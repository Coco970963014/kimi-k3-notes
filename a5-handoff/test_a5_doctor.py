import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import a5_doctor as doctor


class DoctorTests(unittest.TestCase):
    def test_missing_compute_consent(self):
        with mock.patch.object(sys, "argv", ["doctor", "--compute"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                doctor.main()
        self.assertEqual(error.exception.code, 2)

    def test_hidden_probe_requires_consent(self):
        with mock.patch.object(sys, "argv", ["doctor", "--probe", "tensor"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                doctor.main()
        self.assertEqual(error.exception.code, 2)

    def test_subprocess_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for name, script, timeout, expected in (
                ("success", "print('A5_RESULT={\"ok\": true}')", 5, 0),
                ("failure", "raise RuntimeError('TEST_FAILURE')", 5, 1),
                ("timeout", "import time; time.sleep(30)", 0.1, 124),
            ):
                result = doctor.run_stage(name, [sys.executable, "-c", script], root, timeout, os.environ.copy())
                self.assertEqual(result["exit_code"], expected)
                self.assertEqual(result["status"], "PASS" if expected == 0 else "FAIL")
            self.assertIn("TEST_FAILURE", (root / "failure.log").read_text())

    def test_missing_executable(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = doctor.run_stage("missing", [str(Path(directory) / "missing-executable")],
                                      Path(directory), 5, os.environ.copy())
        self.assertEqual(result["exit_code"], 127)

    def run_mocked_main(self, compute=True, failure=None, warning=False):
        called = []

        def fake_stage(name, command, directory, timeout, env):
            called.append(name)
            data = {"packages": {"torch": "2.14.0"}, "warnings": ["test warning"] if warning else []} if name == "inventory" else {}
            return {"status": "FAIL" if name == failure else "WARN" if name == "inventory" and warning else "PASS",
                    "exit_code": 1 if name == failure else 0, "data": data}

        with tempfile.TemporaryDirectory() as directory:
            argv = ["doctor", "--output-root", directory]
            if compute:
                argv += ["--compute", "--device", "0", "--confirm-idle"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "platform", "linux"), \
                    mock.patch.object(doctor, "run_stage", side_effect=fake_stage), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = doctor.main()
            reports = list(Path(directory).glob("*/report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
        return code, report, called

    def test_full_success_scope(self):
        code, report, called = self.run_mocked_main()
        self.assertEqual(code, 0)
        self.assertEqual(report["overall"], "PASS_COMPONENT_SMOKE")
        self.assertEqual(called[-4:], ["tensor", "hccl", "flex", "npu_smi_after"])

    def test_each_compute_failure_stops_later_gates(self):
        for failed in ("imports", "tensor", "hccl", "flex"):
            code, report, called = self.run_mocked_main(failure=failed)
            self.assertEqual(code, 1)
            self.assertEqual(report["overall"], "FAIL")
            sequence = ["imports", "tensor", "hccl", "flex"]
            for later in sequence[sequence.index(failed) + 1:]:
                self.assertNotIn(later, called)
                self.assertEqual(report["stages"][later]["status"], "SKIP")

    def test_inventory_mode_never_computes(self):
        code, report, called = self.run_mocked_main(compute=False)
        self.assertEqual(code, 2)
        self.assertEqual(report["overall"], "INCOMPLETE")
        self.assertNotIn("tensor", called)
        self.assertNotIn("flex", called)

    def test_warnings_not_clean_pass(self):
        code, report, called = self.run_mocked_main(warning=True)
        self.assertEqual(code, 2)
        self.assertEqual(report["overall"], "PASS_WITH_WARNINGS")
        self.assertIn("flex", called)


if __name__ == "__main__":
    unittest.main()
