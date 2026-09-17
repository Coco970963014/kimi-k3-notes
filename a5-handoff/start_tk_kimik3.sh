#!/usr/bin/env bash
set -euo pipefail

# Run on the A5 host: bash start_tk_kimik3.sh YOUR_HOST_USERNAME
# This uses the base image, not packages installed in a colleague's container.
HOST_USER="${1:?Usage: bash start_tk_kimik3.sh YOUR_HOST_USERNAME}"
if [[ "$EUID" -ne 0 ]]; then
    exec sudo -- /bin/bash "$(readlink -f -- "$0")" "$HOST_USER"
fi
NAME=tk-kimik3
IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12

id "$HOST_USER" >/dev/null
HOST_HOME="$(getent passwd "$HOST_USER" | cut -d: -f6)"
[[ "$HOST_HOME" == /* && -d "$HOST_HOME" ]] || { echo 'ERROR: User home not found.' >&2; exit 1; }
WORKSPACE="$HOST_HOME/kimi-workspace"

docker info >/dev/null
if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "ERROR: Container $NAME already exists; nothing changed." >&2
    exit 1
fi
docker image inspect "$IMAGE" >/dev/null || {
    echo "ERROR: Base image not present locally: $IMAGE" >&2
    exit 1
}

devices=()
for node in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc /dev/ummu /dev/uburma; do
    [[ -c "$node" ]] || { echo "ERROR: Missing character device: $node" >&2; exit 1; }
    devices+=(--device "$node")
done

mounts=()
for path in /usr/local/dcmi /usr/local/bin/npu-smi \
    /usr/local/Ascend/driver/lib64 /usr/local/Ascend/driver/version.info \
    /etc/ascend_install.info; do
    [[ -e "$path" ]] || { echo "ERROR: Missing host path: $path" >&2; exit 1; }
    mounts+=(--mount "type=bind,src=$path,dst=$path,readonly")
done

if [[ ! -d "$WORKSPACE" ]]; then
    install -d -o "$HOST_USER" -g "$(id -gn "$HOST_USER")" "$WORKSPACE"
fi

# Eight mapped cards are not reserved; coordinate device use with colleagues.
# Host networking shares ports. Private IPC does not share their /dev/shm.
docker run -dit \
    --name "$NAME" \
    --user 0:0 \
    --network host \
    --ipc private \
    --shm-size 16g \
    "${devices[@]}" \
    "${mounts[@]}" \
    --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
    --workdir /workspace \
    --entrypoint /bin/bash \
    "$IMAGE" -l

printf 'Workspace: %s -> /workspace\n' "$WORKSPACE"
printf 'Enter: sudo docker exec -it --user 0:0 %s /bin/bash -l\n' "$NAME"
