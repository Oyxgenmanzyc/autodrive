#!/usr/bin/env bash
# Always invoke as a child bash; no deletion or overwrite of any older experiment.
set -euo pipefail
COMMIT=${1:?Pass the full release commit}
ARCHIVE=${2:-}
INSTALL_ROOT=${INSTALL_ROOT:-/home/hndx/zyc/training_code/3.1.05}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid full commit'; exit 1; }
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
mkdir -p "$INSTALL_ROOT"
DEST="$INSTALL_ROOT/autodrive-cost-ranked-pcs-3.1.05_6-${COMMIT:0:12}"
if [ -e "$DEST" ]; then
    if [ -f "$DEST/.release_commit" ] && [ "$(cat "$DEST/.release_commit")" = "$COMMIT" ]; then
        echo "Already installed. CODE_ROOT=$DEST"
        exit 0
    fi
    echo 'Destination exists without matching release marker. Choose another INSTALL_ROOT.'
    exit 1
fi
if [ -z "$ARCHIVE" ]; then
    ARCHIVE=$(mktemp "$INSTALL_ROOT/cost_rank_${COMMIT:0:12}_XXXXXX.zip")
    curl --fail --location --retry 2 --connect-timeout 15 --max-time 300 \
        "https://codeload.github.com/Oyxgenmanzyc/autodrive/zip/$COMMIT" -o "$ARCHIVE"
fi
python - "$ARCHIVE" "$DEST" "$COMMIT" <<'PY'
import sys, zipfile
from pathlib import Path, PurePosixPath
archive, destination, commit = sys.argv[1:]
root = Path(destination)
with zipfile.ZipFile(archive) as z:
    members = [m for m in z.infolist() if not m.is_dir()]
    if {PurePosixPath(m.filename).parts[0] for m in members} != {'autodrive-'+commit}:
        raise SystemExit('Archive prefix must match pinned Git commit')
    paths = []
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
            raise SystemExit('Unsafe archive member')
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise SystemExit('Unexpected symlink')
        paths.append((member, Path(*path.parts[1:])))
    if len({p.as_posix() for _, p in paths}) != len(paths):
        raise SystemExit('Duplicate archive member')
    if not any(p.as_posix() == 'scripts/pcs/run_3_1_05_6.sh' for _, p in paths):
        raise SystemExit('Wrong release: no 3.1.05_6 launcher')
    root.mkdir(exist_ok=False)
    for member, relative in paths:
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(z.read(member))
(root/'.release_commit').write_text(commit+'\n')
print('Installed:', root)
print('Git LF bytes preserved. No manual line-ending conversion is required.')
print('CODE_ROOT='+str(root))
PY
