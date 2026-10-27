#!/bin/sh
# Build MBX (MB-pol) as a shared library for E4 (sseft/mbpol.py, notes/e4-plan.md).
#
#   sh external/mbx/build.sh [DEST]          DEST defaults to external/mbx/build; MBX lands in DEST/MBX (MBX_HOME)
#   JOBS=8 sh external/mbx/build.sh          parallel compile (default 4)
#   MBX_REPO=/path/to/MBX sh ...             an existing clone of github.com/paesanilab/MBX instead of cloning
#   FFTW_HOME=/path sh ...                   where FFTW 3 lives (include/fftw3.h, lib/libfftw3.so); found by default in
#                                            $CONDA_PREFIX, $EBROOTFFTW (an EasyBuild module) or the system
#
# What it needs:
#   - g++ 9.3 or newer with OpenMP;
#   - autoconf, automake and libtool (for autoreconf);
#   - FFTW 3 with its headers.
# On Niflheim either `module load FFTW` (EasyBuild) or, in the conda env,
#   conda install -c conda-forge fftw autoconf automake libtool
#
# Run the clone where there is internet access (the login node). The compile takes about 15-30 min on 4-8 cores and
# about an hour on 2: MBX's 2- and 3-body polynomials are some 50 MB of C++.
#
# After installing, the script runs sseft/mbpol.py's check:
#   - MBX's own MB-pol example (a trimer, open and in a 20 A box) to 1e-8 kcal/mol;
#   - the 1-body-free interaction energy;
#   - forces against differences;
#   - the monomer's polarisability through the external field.
# MBX: M. Riera et al., J. Chem. Phys. 159, 054802 (2023); MB-pol: Babin, Leforestier, Paesani, JCTC 9, 5395 (2013),
# 10, 1599 and 2906 (2014). MBX is free for research and non-profit use under the UC licence in its repository.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
DEST=${1:-$HERE/build}
JOBS=${JOBS:-4}
PYTHON=${PYTHON:-python3}
EXPECTED_COMMIT=0e01b75          # MBX v1.4.0, the commit E4 was set up with (27 Oct 2026)

command -v g++ >/dev/null 2>&1 || { echo "error: no g++ here"; exit 1; }
command -v autoreconf >/dev/null 2>&1 || { echo "error: no autoreconf: conda install -c conda-forge autoconf automake libtool"; exit 1; }

# FFTW: the first of FFTW_HOME, CONDA_PREFIX, EBROOTFFTW with include/fftw3.h; else the system's
FFTW=""
for d in "${FFTW_HOME:-}" "${CONDA_PREFIX:-}" "${EBROOTFFTW:-}"; do
  if [ -n "$d" ] && [ -f "$d/include/fftw3.h" ]; then FFTW=$d; break; fi
done
if [ -z "$FFTW" ] && [ ! -f /usr/include/fftw3.h ]; then
  echo "error: no FFTW 3 headers found: module load FFTW, or conda install -c conda-forge fftw, or set FFTW_HOME"; exit 1
fi

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
if [ -n "${MBX_REPO:-}" ]; then
  rm -rf "$DEST/MBX" && cp -R "$MBX_REPO" "$DEST/MBX"
elif [ ! -d "$DEST/MBX" ]; then
  git clone --quiet https://github.com/paesanilab/MBX.git "$DEST/MBX"
fi
cd "$DEST/MBX"
commit=$(git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
[ "$commit" = "$EXPECTED_COMMIT" ] || echo "note: MBX at commit $commit, E4 was set up with $EXPECTED_COMMIT (v1.4.0)"

if [ -n "$FFTW" ]; then
  # MBX's configure reads FFTW_HOME; the rpath lets the library find libfftw3 at run time without LD_LIBRARY_PATH
  export FFTW_HOME="$FFTW"
  export LDFLAGS="${LDFLAGS:-} -Wl,-rpath,$FFTW/lib"
fi
echo "configuring MBX in $DEST/MBX (FFTW: ${FFTW:-system})"
autoreconf -fi > "$DEST/autoreconf.log" 2>&1 || { tail -20 "$DEST/autoreconf.log"; exit 1; }
./configure --enable-shared CXX=g++ > "$DEST/configure.log" 2>&1 || { tail -20 "$DEST/configure.log"; exit 1; }
echo "compiling with $JOBS jobs (log: $DEST/make.log)"
make -j "$JOBS" > "$DEST/make.log" 2>&1 || { tail -30 "$DEST/make.log"; exit 1; }
make install > "$DEST/install.log" 2>&1 || { tail -20 "$DEST/install.log"; exit 1; }
[ -f "$DEST/MBX/lib/libmbx.so" ] || { echo "error: no $DEST/MBX/lib/libmbx.so after make install"; exit 1; }
echo "built $DEST/MBX/lib/libmbx.so"

cd "$REPO_ROOT"
"$PYTHON" sseft/mbpol.py --check "$DEST/MBX"
