#!/bin/sh
# Runs the Swift tests. Offline: nothing here opens the app, the screen, the microphone or the network.
#
# With only the Command Line Tools installed (no Xcode), SwiftPM does not find the Swift Testing macro plugin
# by itself and fails with "plugin for module 'TestingMacros' not found". This script passes its location when
# it exists. With Xcode selected the extra flag is not needed and is skipped.
set -eu
cd "$(dirname "$0")/.."
plugins="$(dirname "$(xcrun --find swift)")/../lib/swift/host/plugins/testing"
if [ -d "$plugins" ] && ! xcode-select -p | grep -q '\.app/'; then
  exec swift test -Xswiftc -plugin-path -Xswiftc "$plugins" "$@"
fi
exec swift test "$@"
