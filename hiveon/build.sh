#!/usr/bin/env bash
# Build the Hiveon custom miner archive.
#
# Hive requires <name>-<version>.tar.gz containing a single top-level directory
# named after the miner, with the h-* scripts executable inside it. The version
# must not contain a hyphen: Hive splits the filename on the last one.
set -euo pipefail
cd "$(dirname "$0")"

VERSION="$(grep -oP '^CUSTOM_VERSION=\K.*' proteus/h-manifest.conf | tr -d '"')"
[[ "$VERSION" == *-* ]] && { echo "version must not contain a hyphen: $VERSION" >&2; exit 1; }

chmod +x proteus/h-config.sh proteus/h-run.sh proteus/h-stats.sh

OUT="proteus-${VERSION}.tar.gz"
rm -f proteus-*.tar.gz
tar -zcf "$OUT" --owner=0 --group=0 proteus

echo "$OUT  $(stat -c%s "$OUT") bytes"
tar -tzf "$OUT"
