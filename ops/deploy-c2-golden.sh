#!/usr/bin/env bash
# Run ON 5090-omarchy, as florious, after the benchmark owner releases the GPU.
set -euo pipefail
umask 077

if [[ $# != 3 ]]; then
  echo "usage: $0 /absolute/frozen-golden.json SHA256 /absolute/qualified-venv/bin/python" >&2
  exit 2
fi
ops=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
expected="$HOME/Projects/Strata-5090-delivery/ops"
[[ "$ops" == "$expected" ]] || { echo "FAIL: delivery checkout must be $expected" >&2; exit 2; }
config="$HOME/.config/strata-c2-golden/production.json"
key_config="$HOME/Projects/Strata/strata-iq3_s.json"
state="$HOME/.local/state/strata-c2-golden"
mkdir -p "$state" "$HOME/.config/systemd/user"
exec 9>"$state/deploy.lock"
flock -n 9 || { echo 'FAIL: another deployment owns the lock' >&2; exit 2; }

# Verify release/config/key/assets before stopping anything. Never print private config contents.
python3 "$ops/c2_golden.py" prepare --golden-config "$1" --golden-config-sha256 "$2" \
  --key-config "$key_config" --python "$3" --output "$config"
"$3" -c 'import sys; sys.path.insert(0,"/home/florious/Projects/Strata-c2-v0.1.43-port"); import serve.server'
python3 "$ops/c2_golden.py" check --config "$config"
[[ "$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("BackendState", ""))')" == Running ]]
[[ "$(tailscale ip -4)" == 100.99.91.50 ]]

# Old official production unit only. Unknown benchmark/desktop/other GPU PIDs are not killed.
systemctl --user stop strata-iq3.service
python3 "$ops/c2_golden.py" check --config "$config" --idle
unit="$HOME/.config/systemd/user/strata-c2-golden.service"
if [[ -e "$unit" ]]; then
  cp -- "$unit" "$state/strata-c2-golden.service.$(date +%Y%m%dT%H%M%S).bak"
fi
install -m 600 "$ops/strata-c2-golden.service" "$unit"
systemctl --user daemon-reload
systemctl --user disable strata-iq3.service
if [[ "$(loginctl show-user "$(id -un)" -p Linger --value)" != yes ]]; then
  loginctl enable-linger "$(id -un)" || sudo -n loginctl enable-linger "$(id -un)"
fi
systemctl --user enable --now strata-c2-golden.service
if ! python3 "$ops/c2_golden.py" smoke --config "$config" --timeout 240; then
  systemctl --user disable --now strata-c2-golden.service
  echo 'FAIL: stopped failed new release; old source/config remain intact. Use documented explicit rollback.' >&2
  exit 3
fi
# Authenticated requests stay on target; no secret is passed to the controller or argv.
python3 "$ops/c2_golden.py" smoke --config "$config" --base http://100.99.91.50:8080 --timeout 120
systemctl --user show strata-c2-golden.service -p ActiveState -p SubState -p MainPID -p NRestarts
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
printf 'local_and_TS_auth_smoke=PASS; controller LAN/TS reachability and native C2 dispatch still require operator acceptance\n'
