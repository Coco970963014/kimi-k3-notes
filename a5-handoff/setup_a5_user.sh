#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

# Run as root on the host: bash setup_a5_user.sh [username]
# Both full sudo and Docker membership effectively grant host root access.
ACCOUNT="${1:-t0897426}"
[[ "$EUID" -eq 0 ]] || { echo 'ERROR: Run as root on the host.' >&2; exit 1; }
[[ ! -e /.dockerenv ]] || { echo 'ERROR: Run on the host, not inside Docker.' >&2; exit 1; }
[[ "$ACCOUNT" =~ ^[a-z_][a-z0-9_-]*$ && "$ACCOUNT" != root ]] || {
    echo 'ERROR: Invalid account name or root account requested.' >&2
    exit 1
}
for cmd in getent useradd usermod passwd sudo visudo install mktemp; do
    command -v "$cmd" >/dev/null || { echo "ERROR: Required command missing: $cmd" >&2; exit 1; }
done
for group in docker HwHiAiUser; do
    getent group "$group" >/dev/null || { echo "ERROR: Required group missing: $group" >&2; exit 1; }
done
[[ -d /etc/sudoers.d ]] || { echo 'ERROR: /etc/sudoers.d does not exist.' >&2; exit 1; }
POLICY="/etc/sudoers.d/90-${ACCOUNT}-admin"

TEMP_POLICY="$(mktemp /etc/sudoers.d/.a5-user-XXXXXX)"
trap 'rm -f -- "$TEMP_POLICY"' EXIT
printf '%s ALL=(ALL:ALL) PASSWD: ALL\n' "$ACCOUNT" > "$TEMP_POLICY"
chmod 0600 "$TEMP_POLICY"
visudo -cf "$TEMP_POLICY"
if [[ -e "$POLICY" || -L "$POLICY" ]]; then
    if [[ -L "$POLICY" ]] || ! cmp -s "$TEMP_POLICY" "$POLICY"; then
        echo "ERROR: Different existing policy at $POLICY; refusing to overwrite it." >&2
        exit 1
    fi
fi

if id "$ACCOUNT" >/dev/null 2>&1; then
    [[ "$(id -u "$ACCOUNT")" -ne 0 ]] || { echo 'ERROR: Refusing a UID 0 account.' >&2; exit 1; }
    echo "Existing account: $ACCOUNT; password, home and shell unchanged."
else
    useradd -m -s /bin/bash "$ACCOUNT"
    echo "Set a strong password for $ACCOUNT (interactive; not stored in this script)."
    passwd "$ACCOUNT"
fi

usermod -aG docker,HwHiAiUser "$ACCOUNT"
install -o root -g root -m 0440 "$TEMP_POLICY" "$POLICY"
id "$ACCOUNT"
grep -Eq '^[[:space:]]*([@#]includedir)[[:space:]]+"?/etc/sudoers.d"?([[:space:]]|$)' /etc/sudoers ||
    echo 'WARNING: Standard sudoers.d include not found; confirm the new rule is loaded.' >&2
if ! sudo -l -U "$ACCOUNT"; then
    echo 'WARNING: Account/groups/rule written, but sudo verification failed; other sudoers files were not changed.' >&2
fi
echo 'Account/groups/rule written. Log out and log in again; verify effective sudo access below.'
echo 'Then verify: id; docker ps; sudo -k; sudo whoami'
