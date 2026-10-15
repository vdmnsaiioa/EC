# Shared by build.sh (q-AQUA v1) and build_pol.sh (q-AQUA-pol), sourced after FC is set: link a shared library so that
# it carries the compiler's run-time libraries, and check that it loads without the compiler's environment.
#
#   link_with_runtime LIB OBJECTS...   link LIB from OBJECTS with $FC $FFLAGS and DT_RPATH $ORIGIN/runtime, copy the
#                                      run-time libraries it needs (libgfortran, and libquadmath where the compiler uses
#                                      it) to <dir of LIB>/runtime, and fail unless ldd resolves every library with
#                                      LD_LIBRARY_PATH unset
#
# The RPATH is a DT_RPATH (--disable-new-dtags): it also serves the runtime's own dependencies (libgfortran ->
# libquadmath) and LD_LIBRARY_PATH does not override it, so the library runs on the runtime it was compiled against.

needed() {      # the libraries an ELF file needs (its DT_NEEDED; with neither readelf nor objdump, ldd's closure)
  if command -v readelf >/dev/null 2>&1; then readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p'
  elif command -v objdump >/dev/null 2>&1; then objdump -p "$1" | awk '$1 == "NEEDED" {print $2}'
  else ldd "$1" 2>/dev/null | awk '$2 == "=>" {print $1}'; fi
}

system_library() {   # the C library's parts and libgcc_s: on every Linux system, never bundled
  case "$1" in
    ld-linux*|ld64.so*|libc.so*|libm.so*|libmvec.so*|libdl.so*|libpthread.so*|librt.so*|libutil.so*|libresolv.so*|libgcc_s.so*) return 0 ;;
  esac
  return 1
}

locate() {      # the file of library $2 needed by $1: the compiler's own (what it linked), else what this shell resolves
  p=$($FC -print-file-name="$2" 2>/dev/null || true)
  case "$p" in /*) if [ -e "$p" ]; then echo "$p"; return 0; fi ;; esac
  ldd "$1" 2>/dev/null | awk -v n="$2" '$1 == n && $3 ~ /^\// {print $3; exit}'
}

link_with_runtime() {
  lib=$1; shift
  rt="$(dirname "$lib")/runtime"
  rm -rf "$rt" && mkdir -p "$rt"
  $FC $FFLAGS -shared -o "$lib" "$@" -Wl,--disable-new-dtags,-rpath,'$ORIGIN/runtime'
  queue=$(needed "$lib")
  seen=" "
  while :; do
    set -- $queue
    [ $# -gt 0 ] || break
    name=$1; shift; queue="$*"
    case "$seen" in *" $name "*) continue ;; esac
    seen="$seen$name "
    if system_library "$name"; then continue; fi
    file=$(locate "$lib" "$name")
    [ -n "$file" ] || { echo "error: cannot find $name (needed by $(basename "$lib")): build where $FC's environment is loaded"; exit 1; }
    cp -L "$file" "$rt/$name"
    echo "  bundled $name ($file)"
    queue="$queue $(needed "$rt/$name")"
  done
  if command -v ldd >/dev/null 2>&1; then
    resolved=$(env -u LD_LIBRARY_PATH ldd "$lib" 2>&1 || true)
    if printf '%s\n' "$resolved" | grep -q "not found"; then
      printf '%s\n' "$resolved"
      echo "error: $(basename "$lib") does not find all its libraries without the compiler's environment"; exit 1
    fi
    printf '%s\n' "$resolved" | grep -F "$rt/" | sed 's/ *(0x[0-9a-f]*)$//' || true
  fi
}
