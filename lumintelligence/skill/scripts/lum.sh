#!/bin/sh
# Runs the LumIntelligence CLI for the Hermes skill, always with JSON output.
# Finds `lum` without relying on the app's PATH: $LUM_BIN, the path recorded by `lum skill install`
# (scripts/.lum_bin), uv's tool bin folder, ~/.local/bin, PATH.
here="$(cd "$(dirname "$0")" && pwd)"
recorded=""
[ -f "$here/.lum_bin" ] && recorded="$(cat "$here/.lum_bin")"
for candidate in "${LUM_BIN:-}" "$recorded" "${UV_TOOL_BIN_DIR:+$UV_TOOL_BIN_DIR/lum}" "${XDG_BIN_HOME:+$XDG_BIN_HOME/lum}" \
                 "$HOME/.local/bin/lum" "$(command -v lum 2>/dev/null)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    exec "$candidate" --json "$@"
  fi
done
printf '%s\n' '{"error": "The lum command was not found.", "hint": "Install it with: uv tool install --from <LumIntelligence repo or URL> \"lumintelligence-engine[openai,files]\", then run `lum init` and `lum skill install` in a terminal. If lum is installed somewhere else, set LUM_BIN to its full path."}'
exit 127
