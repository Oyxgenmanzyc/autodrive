#!/usr/bin/env bash
# Run as a child shell; failure never exits the user's interactive terminal.
set -euo pipefail
COMMIT=${1:?Pass the complete 40-character Git commit}
ARCHIVE=${2:-}
INSTALL_ROOT=${INSTALL_ROOT:-/home/hndx/zyc/training_code/3.1.05}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit'; exit 1; }
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
mkdir -p "$INSTALL_ROOT"
DEST="$INSTALL_ROOT/autodrive-gated-paired-timing-3.1.05_5-${COMMIT:0:12}"
if [ -d "$DEST" ]; then
    if [ -f "$DEST/.release_commit" ] && [ "$(cat "$DEST/.release_commit")" = "$COMMIT" ]; then
        echo "Already installed: $DEST"
        exit 0
    fi
    echo "Destination already exists without matching release marker: $DEST"
    echo 'Use a different INSTALL_ROOT; existing files were not overwritten.'
    exit 1
fi
if [ -z "$ARCHIVE" ]; then
    ARCHIVE=$(mktemp "$INSTALL_ROOT/gated_${COMMIT:0:12}_XXXXXX.zip")
    curl --fail --location --retry 3 --connect-timeout 20 \
        "https://codeload.github.com/Oyxgenmanzyc/autodrive/zip/$COMMIT" -o "$ARCHIVE"
fi
python - "$ARCHIVE" "$DEST" "$COMMIT" <<'PY'
import sys, zipfile
from pathlib import Path, PurePosixPath
archive, destination, commit = sys.argv[1:]
root = Path(destination)
with zipfile.ZipFile(archive) as z:
    members = [m for m in z.infolist() if not m.is_dir()]
    prefixes = {PurePosixPath(m.filename).parts[0] for m in members}
    if len(prefixes) != 1:
        raise SystemExit('Archive must contain one release directory')
    if prefixes != {'autodrive-'+commit}:
        raise SystemExit('Archive root does not match pinned commit')
    paths = []
    for m in members:
        path = PurePosixPath(m.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in m.filename:
            raise SystemExit('Unsafe archive member')
        if (m.external_attr >> 16) & 0o170000 == 0o120000:
            raise SystemExit('Unexpected symlink in release')
        paths.append((m, Path(*path.parts[1:])))
    if not any(str(p).replace('\\', '/') == 'scripts/pcs/run_3_1_05_5.sh' for _, p in paths):
        raise SystemExit('Wrong release: missing 3.1.05_5 launcher')
    root.mkdir(exist_ok=False)
    for m, relative in paths:
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(z.read(m))
(root/'.release_commit').write_text(commit+'\n')
print('Installed:', root)
print('Source files retain Git LF bytes; no CRLF conversion needed.')
PY
echo "CODE_ROOT=$DEST"
