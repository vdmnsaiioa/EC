#!/bin/sh
# Build q-AQUA-pol as a shared library for E3-QP (sseft/qaqua.py QAquaPol, configs/water_qaqua_pol.yaml).
#
#   sh external/qaqua/build_pol.sh [DEST]            DEST defaults to external/qaqua/build_pol
#   sh external/qaqua/build_pol.sh --relink [DEST]   relink the objects of an earlier build (seconds: no download, the
#                                                    potential not recompiled); needs the compiler that built them
#   FC=ifort sh external/qaqua/build_pol.sh          the default compiler is gfortran (conda: conda install -c conda-forge gfortran)
#   QAQUA_POL_REPO=/path/to/q-AQUA-pol sh ...        an existing clone of github.com/jmbowma/q-AQUA-pol instead of cloning
#   QAQUA_REPO=/path/to/q-AQUA sh ...                an existing clone of github.com/jmbowma/q-AQUA (its raw data sets)
#
# Produces DEST/libqaqua_pol.so (q-AQUA-pol's sources + qaqua_pol_capi.f90) with the compiler's run-time libraries in
# DEST/runtime (see build.sh), DEST/q-AQUA-pol (the clone: q-AQUA-pol/src and q-AQUA-pol/coef_qAQUA-pol, from which
# the potential reads its coefficients at run time) and DEST/q-AQUA (a clone of the q-AQUA v1 repository, for its raw
# CCSD(T) 2-, 3- and 4-body sets: the held-out reading of the clusters part, the same frames as E3-Q's).  Run it where
# there is internet access (the login node); the three generated polynomial files take several minutes to compile
# and are compiled in parallel.  q-AQUA-pol: Q. Yu et al., J. Chem. Theory Comput. 19, 3446 (2023), CC0.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
RELINK=0
if [ "${1:-}" = "--relink" ]; then RELINK=1; shift; fi
DEST=${1:-$HERE/build_pol}
FC=${FC:-gfortran}
PYTHON=${PYTHON:-python3}
EXPECTED_COMMIT=3a38dc4          # the q-AQUA-pol commit E3-QP was set up and pre-registered with (2023-06-24)
EXPECTED_DATA_COMMIT=91c90d2     # the q-AQUA v1 commit of the raw sets (E3-Q's)
command -v "$FC" >/dev/null 2>&1 || { echo "error: no $FC here: load the Fortran compiler (module, or the conda env with gfortran)"; exit 1; }
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
OBJ="$DEST/obj"
SRC="$DEST/q-AQUA-pol/q-AQUA-pol/src"
case "$FC" in
  *ifort*|*ifx*) FFLAGS=${FFLAGS:-"-r8 -O2 -fPIC"}; MODDIR="-module $OBJ" ;;
  *)             FFLAGS=${FFLAGS:-"-O2 -fPIC -fdefault-real-8 -fdefault-double-8 -std=legacy -w"}; MODDIR="-J$OBJ" ;;
esac
compile() { $FC $FFLAGS $MODDIR -I"$OBJ" -c "$1" -o "$OBJ/$(basename "$1" .f90).o"; }
POTENTIAL="constants model_mod ttm2f_mod ttm3f_mod nasa_mod math smear nasa potential_mod bemsa2b bemsa3b bemsa4b pes_shell"

clone() {       # clone() DIR URL OVERRIDE EXPECTED: the repository at DEST/DIR (OVERRIDE: a local clone to copy)
  if [ -n "$3" ]; then
    rm -rf "$DEST/$1" && cp -R "$3" "$DEST/$1"
  elif [ ! -d "$DEST/$1" ]; then
    git clone --quiet "$2" "$DEST/$1"
  fi
  commit=$(cd "$DEST/$1" && git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
  [ "$commit" = "$4" ] || echo "note: $1 at commit $commit, E3-QP was set up with $4"
}

if [ "$RELINK" = 1 ]; then
  for f in $POTENTIAL; do
    [ -f "$OBJ/$f.o" ] || { echo "error: no $OBJ/$f.o to relink: run sh external/qaqua/build_pol.sh without --relink"; exit 1; }
  done
  [ -d "$DEST/q-AQUA-pol/q-AQUA-pol/coef_qAQUA-pol" ] || { echo "error: no coefficients under $DEST/q-AQUA-pol: run without --relink"; exit 1; }
  echo "relinking the q-AQUA-pol objects of $OBJ with $FC"
else
  # 1. the repositories: the potential, and q-AQUA v1 for the raw CCSD(T) sets
  clone q-AQUA-pol https://github.com/jmbowma/q-AQUA-pol.git "${QAQUA_POL_REPO:-}" "$EXPECTED_COMMIT"
  clone q-AQUA https://github.com/jmbowma/q-AQUA.git "${QAQUA_REPO:-}" "$EXPECTED_DATA_COMMIT"

  # 2. compile: their default REAL is 8 bytes (their Makefile's MKL serves only the normal-mode code, not built here;
  #    their OpenMP directives are left off: the library is called from one thread)
  rm -rf "$OBJ" && mkdir -p "$OBJ"
  echo "compiling q-AQUA-pol with $FC $FFLAGS"
  for f in constants model_mod ttm2f_mod ttm3f_mod nasa_mod math smear nasa potential_mod; do compile "$SRC/$f.f90"; done
  for f in bemsa2b bemsa3b bemsa4b; do compile "$SRC/$f.f90" & done
  wait
  for f in bemsa2b bemsa3b bemsa4b; do [ -f "$OBJ/$f.o" ] || { echo "compiling $f.f90 failed"; exit 1; }; done
  compile "$SRC/pes_shell.f90"
fi
compile "$HERE/qaqua_pol_capi.f90"

# 3. link, with the compiler's run-time libraries in DEST/runtime (external/qaqua/runtime.sh)
. "$HERE/runtime.sh"
link_with_runtime "$DEST/libqaqua_pol.so" "$OBJ"/*.o
echo "built $DEST/libqaqua_pol.so"

# 4. the authors' benchmark (README of q-AQUA-pol: the hexamer, energy and gradient) and TTM3-F's long range
"$PYTHON" "$REPO_ROOT/sseft/qaqua.py" --check "$DEST"
