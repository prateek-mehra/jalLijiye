#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script only supports macOS (launchd)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${JAL_LAUNCH_LABEL:-com.prateek.jallijiye}"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$ROOT_DIR/logs/startup"
UID_VALUE="$(id -u)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
RUNNER_SCRIPT="$LOG_DIR/run_jallijiye.sh"
APP_BUNDLE="$ROOT_DIR/dist/JalLijiye.app"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python3 not found. Install Python or create .venv first." >&2
    exit 1
  fi
fi

xml_escape() {
  local raw="${1//&/&amp;}"
  raw="${raw//</&lt;}"
  raw="${raw//>/&gt;}"
  printf '%s' "$raw"
}

shell_join() {
  local out=""
  local arg=""
  for arg in "$@"; do
    out+=$(printf '%q ' "$arg")
  done
  printf '%s' "${out% }"
}

build_runner_script() {
  local -a full_cmd=("$PYTHON_BIN" -m app.main)
  mkdir -p "$LOG_DIR"

  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf 'cd %q\n' "$ROOT_DIR"
    printf 'while true; do\n'
    printf '  %s >> %q 2>> %q\n' \
      "$(shell_join "${full_cmd[@]}")" \
      "$LOG_DIR/stdout.log" \
      "$LOG_DIR/stderr.log"
    printf '  sleep 2\n'
    printf 'done\n'
  } > "$RUNNER_SCRIPT"
  chmod +x "$RUNNER_SCRIPT"
}

build_plist() {
  mkdir -p "$HOME/Library/LaunchAgents"
  mkdir -p "$LOG_DIR"

  {
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "$LABEL")</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>WorkingDirectory</key>
  <string>$(xml_escape "$ROOT_DIR")</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/open</string>
    <string>$(xml_escape "$APP_BUNDLE")</string>
  </array>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$LOG_DIR/stdout.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$LOG_DIR/stderr.log")</string>
</dict>
</plist>
EOF
  } > "$PLIST_PATH"
}

build_terminal_fallback_plist() {
  local runner_cmd="/bin/bash $(printf '%q' "$RUNNER_SCRIPT")"
  mkdir -p "$HOME/Library/LaunchAgents"
  mkdir -p "$LOG_DIR"

  {
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "$LABEL")</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>WorkingDirectory</key>
  <string>$(xml_escape "$ROOT_DIR")</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/osascript</string>
    <string>-e</string>
    <string>tell application "Terminal" to do script "$(xml_escape "$runner_cmd")"</string>
    <string>-e</string>
    <string>tell application "Terminal" to hide</string>
  </array>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$LOG_DIR/stdout.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$LOG_DIR/stderr.log")</string>
</dict>
</plist>
EOF
  } > "$PLIST_PATH"
}

stop_job() {
  launchctl bootout "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 || true
  pkill -f "$RUNNER_SCRIPT" >/dev/null 2>&1 || true
}

start_job() {
  if [[ ! -f "$PLIST_PATH" ]]; then
    echo "LaunchAgent not installed: $PLIST_PATH" >&2
    exit 1
  fi
  stop_job
  launchctl bootstrap "gui/${UID_VALUE}" "$PLIST_PATH"
  launchctl enable "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1 || true
}

status_job() {
  if launchctl print "gui/${UID_VALUE}/${LABEL}" >/dev/null 2>&1; then
    echo "JalLijiye autostart is loaded (${LABEL})."
    launchctl print "gui/${UID_VALUE}/${LABEL}" | sed -n '1,25p'
    if pgrep -fl "$RUNNER_SCRIPT" >/dev/null 2>&1; then
      echo ""
      echo "Runner process:"
      pgrep -fl "$RUNNER_SCRIPT" | sed -n '1,5p'
    else
      echo "Runner process not found."
    fi
  else
    echo "JalLijiye autostart is not loaded (${LABEL})."
    if [[ -f "$PLIST_PATH" ]]; then
      echo "Plist exists at: $PLIST_PATH"
    fi
  fi
}

usage() {
  cat <<EOF
Usage:
  scripts/autostart_macos.sh install
  scripts/autostart_macos.sh start
  scripts/autostart_macos.sh stop
  scripts/autostart_macos.sh status
  scripts/autostart_macos.sh uninstall

Notes:
  - install creates ~/Library/LaunchAgents/${LABEL}.plist
  - If dist/JalLijiye.app exists, autostart launches the app bundle.
  - Otherwise it falls back to a hidden Terminal-hosted runner.
  - The Terminal runner auto-restarts the app every 2 seconds if it exits.
EOF
}

cmd="${1:-}"
case "$cmd" in
  install)
    build_runner_script
    if [[ -d "$APP_BUNDLE" ]]; then
      build_plist
    else
      build_terminal_fallback_plist
    fi
    start_job
    echo "Installed and started LaunchAgent: $PLIST_PATH"
    ;;
  start)
    start_job
    echo "Started LaunchAgent: $PLIST_PATH"
    ;;
  stop)
    stop_job
    echo "Stopped LaunchAgent: $LABEL"
    ;;
  status)
    status_job
    ;;
  uninstall)
    stop_job
    rm -f "$PLIST_PATH"
    echo "Removed LaunchAgent: $PLIST_PATH"
    ;;
  *)
    usage
    exit 1
    ;;
esac
