#!/bin/sh
# Build q-AQUA v1 as a shared library for the E3-Q experiment (sseft/qaqua.py, configs/water_qaqua.yaml).
#
#   sh external/qaqua/build.sh [DEST]            DEST defaults to external/qaqua/build
#   sh external/qaqua/build.sh --relink [DEST]   relink the objects of an earlier build (seconds: no download, q-AQUA
#                                                not recompiled); needs the compiler that built them
#   FC=ifort sh external/qaqua/build.sh          the default compiler is gfortran (conda: conda install -c conda-forge gfortran)
#   QAQUA_REPO=/path/to/q-AQUA sh ...            use an existing clone of github.com/jmbowma/q-AQUA instead of cloning
#
# Produces DEST/libqaqua.so (the q-AQUA sources + qaqua_capi.f90), DEST/qAQUA_v1/{src,coef} (q-AQUA reads its
# coefficients from there at run time) and DEST/q-AQUA/{2b,3b,4b}_data.xyz (the raw CCSD(T) n-body sets, held out).
# The compiler's run-time libraries that libqaqua.so needs (libgfortran, and libquadmath where that compiler uses it)
# are copied to DEST/runtime, which the library searches first ($ORIGIN/runtime in its RPATH): the build loads in any
# environment on the same OS -- a batch job, another conda env -- without the compiler's module or env loaded.
# Run it where there is internet access (the login node); the four generated polynomial files take several minutes
# to compile and are compiled in parallel.  q-AQUA: Q. Yu et al., J. Phys. Chem. Lett. 13, 5068 (2022), MIT licence.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
RELINK=0
if [ "${1:-}" = "--relink" ]; then RELINK=1; shift; fi
DEST=${1:-$HERE/build}
FC=${FC:-gfortran}
PYTHON=${PYTHON:-python3}
EXPECTED_COMMIT=91c90d2      # the q-AQUA commit E3-Q was set up and pre-registered with (2023-02-28)
command -v "$FC" >/dev/null 2>&1 || { echo "error: no $FC here: load the Fortran compiler (module, or the conda env with gfortran)"; exit 1; }
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
OBJ="$DEST/obj"
SRC="$DEST/qAQUA_v1/src"
case "$FC" in
  *ifort*|*ifx*) FFLAGS=${FFLAGS:-"-r8 -O2 -fPIC"}; MODDIR="-module $OBJ" ;;
  *)             FFLAGS=${FFLAGS:-"-O2 -fPIC -fdefault-real-8 -fdefault-double-8 -std=legacy -w"}; MODDIR="-J$OBJ" ;;
esac
compile() { $FC $FFLAGS $MODDIR -I"$OBJ" -c "$1" -o "$OBJ/$(basename "$1" .f90).o"; }

if [ "$RELINK" = 1 ]; then
  # the q-AQUA objects and data of an earlier build; the C interface is recompiled (a compiler other than the one of
  # those objects stops there, at their module files)
  for f in constants pot_monomer_mod pot_monomer ltp2011_chg ltp2011_dms bemsa2b_v8 bemsa3b_v1 bemsa3b_v9 bemsa4b_v3 pes_shell; do
    [ -f "$OBJ/$f.o" ] || { echo "error: no $OBJ/$f.o to relink: run sh external/qaqua/build.sh without --relink"; exit 1; }
  done
  [ -d "$DEST/qAQUA_v1/coef" ] || { echo "error: no $DEST/qAQUA_v1/coef: run without --relink"; exit 1; }
  echo "relinking the q-AQUA objects of $OBJ with $FC"
else
  # 1. the repository
  if [ -n "${QAQUA_REPO:-}" ]; then
    rm -rf "$DEST/q-AQUA" && cp -R "$QAQUA_REPO" "$DEST/q-AQUA"
  elif [ ! -d "$DEST/q-AQUA" ]; then
    git clone --quiet https://github.com/jmbowma/q-AQUA.git "$DEST/q-AQUA"
  fi
  COMMIT=$(cd "$DEST/q-AQUA" && git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
  [ "$COMMIT" = "$EXPECTED_COMMIT" ] || echo "note: q-AQUA at commit $COMMIT, E3-Q was set up with $EXPECTED_COMMIT"

  # 2. the sources (the zip also carries the authors' ifort objects and module files; nothing of that is used)
  rm -rf "$DEST/qAQUA_v1"
  "$PYTHON" - "$DEST/q-AQUA/qAQUA_v1.zip" "$DEST" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    keep = [m for m in z.namelist() if m.startswith("qAQUA_v1/") and not m.endswith((".o", ".mod", ".x"))]
    z.extractall(sys.argv[2], keep)
EOF

  # 3. compile: their default REAL is 8 bytes
  rm -rf "$OBJ" && mkdir -p "$OBJ"
  echo "compiling q-AQUA with $FC $FFLAGS"
  for f in constants pot_monomer_mod pot_monomer ltp2011_chg ltp2011_dms; do compile "$SRC/$f.f90"; done
  for f in bemsa2b_v8 bemsa3b_v1 bemsa3b_v9 bemsa4b_v3; do compile "$SRC/$f.f90" & done
  wait
  for f in bemsa2b_v8 bemsa3b_v1 bemsa3b_v9 bemsa4b_v3; do [ -f "$OBJ/$f.o" ] || { echo "compiling $f.f90 failed"; exit 1; }; done
  compile "$SRC/pes_shell.f90"
fi
compile "$HERE/qaqua_capi.f90"

# 4. link, with the compiler's run-time libraries copied to DEST/runtime and found there through the RPATH; the
#    library must then resolve every library it needs without the compiler's environment (external/qaqua/runtime.sh)
. "$HERE/runtime.sh"
link_with_runtime "$DEST/libqaqua.so" "$OBJ"/*.o
echo "built $DEST/libqaqua.so"

# 5. the authors' benchmarks (README of qAQUA_v1: the trimer and the prism hexamer, energies and gradients)
"$PYTHON" "$REPO_ROOT/sseft/qaqua.py" --check "$DEST"
