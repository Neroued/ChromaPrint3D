#!/usr/bin/env bash
set -euo pipefail

mkdir -p pkg/lib
cp build/bin/chromaprint3d_server pkg/
find build/_deps/onnxruntime-src/lib -name "libonnxruntime*.dylib*" \
  -exec cp {} pkg/lib/ \; 2>/dev/null || true

export BREW_PREFIX="$(brew --prefix)"
export OPENCV_PREFIX="$(brew --prefix opencv)"
export LIBOMP_PREFIX="$(brew --prefix libomp)"
export JPEG_PREFIX="$(brew --prefix jpeg-turbo)"
export OPENBLAS_PREFIX="$(brew --prefix openblas 2>/dev/null || true)"
export GCC_PREFIX="$(brew --prefix gcc 2>/dev/null || true)"

if [[ -n "${OPENBLAS_PREFIX}" && -d "${OPENBLAS_PREFIX}/lib" ]]; then
  find "${OPENBLAS_PREFIX}/lib" -name "libopenblas*.dylib*" \
    -exec cp {} pkg/lib/ \; 2>/dev/null || true
fi

python3 - <<'PYEOF'
import os
import shutil
import subprocess
from pathlib import Path

pkg = Path("pkg")
lib_dir = pkg / "lib"
initial_dylibs = sorted(
    p for p in lib_dir.iterdir() if p.is_file() and ".dylib" in p.name
)
queue = [pkg / "chromaprint3d_server"] + initial_dylibs
seen = set()

search_dirs = []
for key in ("BREW_PREFIX", "OPENCV_PREFIX", "LIBOMP_PREFIX", "JPEG_PREFIX", "OPENBLAS_PREFIX", "GCC_PREFIX"):
    value = os.environ.get(key)
    if value:
        search_dirs.append(Path(value) / "lib")
search_dirs.append(Path("build/_deps/onnxruntime-src/lib").resolve())
search_dirs.extend([Path("/usr/local/lib"), Path("/opt/homebrew/lib")])
search_dirs = [p for p in search_dirs if p.exists()]


def resolve_dep(dep: str, owner: Path):
    if dep.startswith("@loader_path/"):
        candidate = (owner.parent / dep[len("@loader_path/"):]).resolve()
        if candidate.exists():
            return candidate
        # Homebrew gcc runtime dylibs (e.g. libgcc_s.1.1.dylib) reference
        # siblings relative to the original Cellar layout, which no longer
        # resolves after relocation into the bundle. Fall back to a basename
        # search in the known lib dirs, then the gcc keg tree.
        base = os.path.basename(dep)
        for directory in search_dirs:
            fallback = (Path(directory) / base).resolve()
            if fallback.exists():
                return fallback
        gcc_lib = os.environ.get("GCC_PREFIX")
        if gcc_lib:
            gcc_tree = Path(gcc_lib) / "lib" / "gcc"
            if gcc_tree.exists():
                hits = sorted(gcc_tree.rglob(base))
                if hits:
                    return hits[0]
        return None
    if dep.startswith("@executable_path/"):
        candidate = (pkg / dep[len("@executable_path/"):]).resolve()
        return candidate if candidate.exists() else None
    if dep.startswith("@rpath/"):
        base = os.path.basename(dep)
        for directory in [owner.parent, *search_dirs]:
            candidate = (Path(directory) / base).resolve()
            if candidate.exists():
                return candidate
        return None
    candidate = Path(dep)
    return candidate.resolve() if candidate.exists() else None


while queue:
    owner = queue.pop(0)
    if not owner.exists():
        continue
    owner_key = str(owner.resolve())
    if owner_key in seen:
        continue
    seen.add(owner_key)

    output = subprocess.check_output(["otool", "-L", str(owner)], text=True)
    for raw in output.splitlines()[1:]:
        dep = raw.strip().split(" (", 1)[0]
        if not dep or dep.startswith("/System/") or dep.startswith("/usr/lib/"):
            continue
        source = resolve_dep(dep, owner)
        if source is None:
            continue
        # Use the dep basename (what the binary references) as the destination
        # filename, NOT source.name (the resolved symlink target). Homebrew
        # symlinks like libopencv_imgcodecs.413.dylib resolve to
        # libopencv_imgcodecs.4.13.0.dylib; the bash install_name_tool loop
        # later needs the dep basename to match and rewrite correctly.
        dep_base = os.path.basename(dep)
        dest = lib_dir / dep_base
        if not dest.exists():
            shutil.copy2(source, dest)
        queue.append(dest)

print("Bundled macOS runtime libraries:")
for p in sorted(lib_dir.iterdir()):
    if p.is_file():
        print(" -", p.name)
PYEOF

for file in pkg/chromaprint3d_server pkg/lib/*.dylib*; do
  [ -f "$file" ] || continue
  while IFS= read -r dep; do
    base="$(basename "$dep")"
    if [ -f "pkg/lib/$base" ]; then
      install_name_tool -change "$dep" "@rpath/$base" "$file" || true
    fi
  done < <(otool -L "$file" | awk 'NR>1 {print $1}')
done

install_name_tool -add_rpath "@executable_path/lib" pkg/chromaprint3d_server || true
for lib in pkg/lib/*.dylib*; do
  [ -f "$lib" ] || continue
  base="$(basename "$lib")"
  install_name_tool -id "@rpath/$base" "$lib" || true
  install_name_tool -add_rpath "@loader_path" "$lib" || true
done

cat > pkg/run_chromaprint3d_server.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="${SCRIPT_DIR}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
exec "${SCRIPT_DIR}/chromaprint3d_server" "$@"
EOF
chmod +x pkg/run_chromaprint3d_server.sh

echo "Packaged macOS files:"
ls -lh pkg/
ls -lh pkg/lib/

echo "Verifying no absolute Homebrew/system paths remain..."
verify_fail=0
for file in pkg/chromaprint3d_server pkg/lib/*.dylib*; do
  [ -f "$file" ] || continue
  while IFS= read -r dep; do
    case "$dep" in /usr/local/*|/opt/homebrew/*|/opt/local/*)
      echo "ERROR: $(basename "$file") -> $dep"
      verify_fail=1 ;;
    esac
  done < <(otool -L "$file" | awk 'NR>1 {print $1}')
done
if [ "$verify_fail" -ne 0 ]; then
  echo "::error::Absolute Homebrew/system library paths detected in packaged binaries"
  exit 1
fi
echo "All dylib references are portable."
