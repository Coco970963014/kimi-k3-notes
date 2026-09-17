#!/usr/bin/env python3
"""Offline, non-installing A5 diagnostics. Raw artifacts must stay inside A5."""

import argparse
import datetime
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback


PACKAGES = (
    "torch", "torch-npu", "triton", "triton-ascend", "torchtitan",
    "torchvision", "torchao", "torchdata", "flash-linear-attention",
)
MARKER = "A5_RESULT="


def inventory():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    cann = {}
    roots = [Path("/usr/local/Ascend"), Path.home() / "Ascend"]
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        if os.environ.get(key):
            root = Path(os.environ[key])
            roots.extend((root, root.parent))
    for root in roots:
        for pattern in (
            "version.cfg", "version.info", "*-version.info",
            "ascend-toolkit/latest/version.cfg",
            "ascend-toolkit/latest/*-version.info",
            "ascend-toolkit/latest/*/ascend_toolkit_install.info",
            "driver/version.info", "firmware/version.info",
        ):
            for path in root.glob(pattern):
                if path.is_file():
                    try:
                        cann[str(path)] = path.read_text(errors="replace")[:16384]
                    except OSError as exc:
                        cann[str(path)] = str(exc)
    warnings = []
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        warnings.append("Python is outside the v0.3.0 validated 3.11/3.12 matrix")
    if versions["torch"].split("+")[0] != "2.14.0":
        warnings.append("torch differs from the v0.3.0 validated version 2.14.0")
    if all(versions[p] != "NOT_INSTALLED" for p in ("triton", "triton-ascend")):
        warnings.append("Both Triton distributions exist; inspect file ownership")
    if not cann:
        warnings.append("CANN/driver/firmware version files not found; verify manually")
    return {
        "python": platform.python_version(), "executable": sys.executable,
        "arch": platform.machine(), "os": platform.platform(),
        "packages": versions, "version_files": cann, "warnings": warnings,
        "visibility": {key: os.environ.get(key, "UNSET") for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
    }


def imports():
    result = {}
    errors = []
    for name in ("torch", "torch_npu", "triton"):
        try:
            module = importlib.import_module(name)
            result[name] = {
                "version": getattr(module, "__version__", "UNKNOWN"),
                "file": getattr(module, "__file__", "UNKNOWN"),
            }
        except Exception as exc:
            traceback.print_exc()
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if not errors:
        import torch
        from torch.nn.attention.flex_attention import flex_attention
        from torch.distributed.fsdp import fully_shard
        result["apis"] = {
            "compile": callable(torch.compile), "flex_attention": callable(flex_attention),
            "fully_shard": callable(fully_shard),
        }
    result["errors"] = errors
    if errors:
        result["failed"] = True
    return result


def compute(stage, index):
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")
    count = torch.npu.device_count()
    if not 0 <= index < count:
        raise RuntimeError(f"Logical device {index} out of range; visible count={count}")
    torch.npu.set_device(index)
    device = torch.device(f"npu:{index}")
    result = {"device": str(device), "device_name": torch.npu.get_device_name(index)}
    if stage == "tensor":
        x = torch.arange(8, device=device, dtype=torch.float32)
        torch.testing.assert_close((x * 2).cpu(), torch.arange(8).float() * 2)
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        output = a @ b
        output.float().square().mean().backward()
        torch.npu.synchronize()
        finite = all(bool(torch.isfinite(t).all().item()) for t in (output, a.grad, b.grad))
        if not finite:
            raise RuntimeError("BF16 matmul output/gradients are non-finite")
        result.update(bf16_matmul_backward="PASS", finite=True)
    elif stage == "hccl":
        import torch.distributed as dist
        timeout = datetime.timedelta(seconds=60)
        # Port 0 lets TCPStore bind an unused loopback port without a probe/bind race.
        store = dist.TCPStore("127.0.0.1", 0, 1, True, timeout)
        try:
            dist.init_process_group("hccl", store=store, rank=0, world_size=1, timeout=timeout)
            x = torch.tensor([3.0], device=device)
            dist.all_reduce(x)
            torch.npu.synchronize()
            if x.item() != 3.0:
                raise RuntimeError("HCCL world-size 1 all-reduce mismatch")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        result.update(world_size=1, all_reduce="PASS", teardown="PASS")
    elif stage == "flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        import torch._dynamo

        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.disable = False
        torch.manual_seed(1)
        shape = (1, 2, 128, 64)
        q, k, v = [torch.randn(shape, device=device, dtype=torch.bfloat16,
                              requires_grad=True) for _ in range(3)]

        def causal(batch, head, query, key):
            return query >= key

        mask = create_block_mask(causal, 1, 2, 128, 128, device=device)
        compiled = torch.compile(flex_attention, backend="inductor", fullgraph=True, dynamic=False)
        out = compiled(q, k, v, block_mask=mask)
        loss = out.float().square().mean()
        loss.backward()
        torch.npu.synchronize()
        finite = {name: t is not None and bool(torch.isfinite(t).all().item())
                  for name, t in (("output", out), ("loss", loss),
                                  ("q_grad", q.grad), ("k_grad", k.grad), ("v_grad", v.grad))}
        if not all(finite.values()):
            raise RuntimeError(f"Non-finite Flex output/gradients: {finite}")
        result.update(shape=shape, dtype="bfloat16", finite=finite, loss=loss.item(),
                      backend="inductor", attention="native_flex", fullgraph=True)
    return result


def run_stage(name, command, directory, timeout, env):
    log = directory / f"{name}.log"
    start = time.monotonic()
    timed_out = False
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=(os.name == "posix"))
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
                if isinstance(exc, KeyboardInterrupt):
                    raise
                timed_out = True
                code = 124
        except OSError as exc:
            stream.write(f"{type(exc).__name__}: {exc}\n")
            code = 127
    data = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith(MARKER):
                data = json.loads(line[len(MARKER):])
    result = {"status": "PASS" if code == 0 else "FAIL", "exit_code": code,
              "timeout": timed_out, "seconds": round(time.monotonic() - start, 2),
              "log": log.name, "data": data}
    if code == 0 and data and data.get("warnings"):
        result["status"] = "WARN"
    print(f"{name}: {result['status']} exit={code}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", action="store_true", help="Run tensor, HCCL and Flex gates")
    parser.add_argument("--device", type=int, help="Logical NPU index inside CURRENT visibility mapping")
    parser.add_argument("--confirm-idle", action="store_true", help="Confirm device is authorized, idle and reserved")
    parser.add_argument("--repo", type=Path, help="Optional existing TorchTitan checkout; never modified")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/a5-doctor"))
    parser.add_argument("--timeout", type=int, default=180, help="Seconds per ordinary subprocess")
    parser.add_argument("--flex-timeout", type=int, default=600)
    parser.add_argument("--probe", choices=("inventory", "imports", "tensor", "hccl", "flex"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0 or args.flex_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.device is not None and args.device < 0:
        parser.error("Device must be nonnegative")
    if args.compute and (args.device is None or not args.confirm_idle):
        parser.error("--compute requires --device N --confirm-idle; never assume a shared NPU is idle")
    if args.probe:
        if args.probe in ("tensor", "hccl", "flex") and not args.compute:
            parser.error("Compute probes require --compute and explicit device confirmation")
        try:
            if args.probe == "inventory":
                data = inventory()
            elif args.probe == "imports":
                data = imports()
            else:
                data = compute(args.probe, args.device)
            print(MARKER + json.dumps(data, ensure_ascii=True), flush=True)
            return 1 if data.get("failed") else 0
        except Exception:
            traceback.print_exc()
            return 1
    if args.compute and sys.platform != "linux":
        parser.error("Compute mode requires Linux with Ascend NPU support")

    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.datetime.now().strftime("a5-%Y%m%d-%H%M%S-"),
                                      dir=args.output_root)).resolve()
    directory.chmod(0o700)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["TRITON_CACHE_DIR"] = str(directory / "triton-cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(directory / "inductor-cache")
    env["TORCH_COMPILE_DEBUG_DIR"] = str(directory / "compile-debug")
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "0"
    env["TORCHDYNAMO_DISABLE"] = "0"
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    stages = {}
    print(f"Artifacts (LOCAL ONLY): {directory}", flush=True)

    def probe(name, timeout=None):
        command = [sys.executable, str(Path(__file__).resolve()), "--probe", name]
        if name in ("tensor", "hccl", "flex"):
            command += ["--compute", "--device", str(args.device), "--confirm-idle"]
        stages[name] = run_stage(name, command, directory, timeout or args.timeout, env)

    probe("inventory")
    stages["npu_smi_before"] = run_stage("npu_smi_before", ["npu-smi", "info"], directory, 30, env)
    probe("imports")
    if args.repo:
        for name, tail in (
            ("repo_head", ["rev-parse", "HEAD"]),
            ("repo_v030", ["rev-parse", "v0.3.0^{commit}"]),
            ("repo_status", ["status", "--short"]),
        ):
            stages[name] = run_stage(name, ["git", "-C", str(args.repo), *tail], directory, 30, env)
        if all(stages[n]["exit_code"] == 0 for n in ("repo_head", "repo_v030", "repo_status")):
            exact = (directory / "repo_head.log").read_text().strip() == (directory / "repo_v030.log").read_text().strip()
            clean = not (directory / "repo_status.log").read_text().strip()
            stages["repo_baseline"] = {"status": "PASS" if exact and clean else "WARN",
                                       "exact_v030": exact, "clean": clean}
    else:
        stages["repo_baseline"] = {"status": "SKIP", "reason": "No --repo supplied"}
    ready = all(stages[n]["status"] != "FAIL" for n in ("inventory", "npu_smi_before", "imports"))
    for name in ("tensor", "hccl", "flex"):
        if not args.compute or not ready:
            stages[name] = {"status": "SKIP", "reason": "Compute not authorized or previous gate failed"}
            continue
        probe(name, args.flex_timeout if name == "flex" else args.timeout)
        ready = stages[name]["status"] == "PASS"
    if args.compute:
        stages["npu_smi_after"] = run_stage("npu_smi_after", ["npu-smi", "info"], directory, 30, env)

    failed = [name for name, result in stages.items() if result["status"] == "FAIL"]
    warnings = [name for name, result in stages.items() if result["status"] == "WARN"]
    overall = "FAIL" if failed else ("INCOMPLETE" if not args.compute else
                                     "PASS_WITH_WARNINGS" if warnings else "PASS_COMPONENT_SMOKE")
    summary = [f"OVERALL={overall}", "SCOPE=component smoke only; NOT model/FSDP2 or A5 matrix certification",
               f"FIRST_FAILURE={failed[0] if failed else 'NONE'}"]
    inv = stages["inventory"].get("data") or {}
    summary.append(f"Python={inv.get('python', 'UNKNOWN')}")
    summary.extend(f"{name}={version}" for name, version in inv.get("packages", {}).items())
    loaded = stages["imports"].get("data") or {}
    for name in ("torch", "torch_npu", "triton"):
        summary.append(f"loaded_{name}={loaded.get(name, {}).get('version', 'UNKNOWN')}")
    tensor = stages["tensor"].get("data") or {}
    summary.append(f"NPU={tensor.get('device_name', 'NOT_TESTED')}")
    summary.extend(f"WARNING={warning}" for warning in inv.get("warnings", []))
    summary.extend(f"{name}={result['status']} exit={result.get('exit_code', 'NA')}"
                   for name, result in stages.items())
    summary += ["MANUAL=verify CANN/driver/firmware matrix, device health, occupancy and teardown processes",
                "NOT_TESTED=TorchTitan imports/unit tests, Kimi-K3 training, multi-rank FSDP2, numerical parity",
                "PRIVACY=raw logs stay inside A5; report only versions, gate results and first error keywords"]
    (directory / "report.json").write_text(json.dumps({"overall": overall, "stages": stages}, indent=2), encoding="utf-8")
    (directory / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary))
    print(f"LOCAL_ARTIFACTS={directory}")
    return 1 if failed else (2 if not args.compute or warnings else 0)


if __name__ == "__main__":
    raise SystemExit(main())
