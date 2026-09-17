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
grep -Eq '^[[:space:]]*([@#]includedir)[[:space:]]+"?/etc/sudoers.d"?([[:space:]]|$)' /etc/sudoers || {
    echo 'ERROR: /etc/sudoers must already include /etc/sudoers.d; no policy changed.' >&2
    exit 1
}
visudo -c
POLICY="/etc/sudoers.d/90-${ACCOUNT}-admin"
[[ ! -e "$POLICY" && ! -L "$POLICY" ]] || { echo "ERROR: Policy already exists: $POLICY" >&2; exit 1; }

TEMP_POLICY="$(mktemp /etc/sudoers.d/.a5-user-XXXXXX)"
trap 'rm -f -- "$TEMP_POLICY"' EXIT
printf '%s ALL=(ALL:ALL) PASSWD: ALL\n' "$ACCOUNT" > "$TEMP_POLICY"
chmod 0600 "$TEMP_POLICY"
visudo -cf "$TEMP_POLICY"

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
visudo -c
id "$ACCOUNT"
sudo -l -U "$ACCOUNT"
echo 'Account permissions configured. Log out and log in again to refresh groups.'
echo 'Then verify: id; docker ps; sudo -k; sudo whoami'
