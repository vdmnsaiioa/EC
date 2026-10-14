"""Dispatch a YAML-selected experiment; remaining flags go to its script.

A water configuration that lists several parts (experiment.part: [dimer, clusters]) runs them in sequence, each
with its own output and cache under --out_dir (default results/<config name>/: <part>.json, <part>_cache.pkl);
every part's configuration is checked first (--validate_only), so that a mistake in the second part is reported
before the first one runs.  --part selects one part of such a configuration."""
import argparse
import os
import runpy
import sys
import traceback
from pathlib import Path
import yaml
from experiment_config import config_parts, load_config

SCRIPTS = {"dimer_ladder": "dimer_ladder.py", "water": "e3_water.py", "x2b_toy": "x2b_toy_regression.py"}


def _run(target, config, argv):
    sys.argv = [str(target), "--config", config, *argv]
    runpy.run_path(str(target), run_name="__main__")


def _given(remaining, *options):
    return any(token.split("=", 1)[0] in options for token in remaining)


def main():
    # no prefix matching: --out (the script's) must not be read as an abbreviation of --out_dir
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None, help="outputs of every part: <out_dir>/<part>.json and <part>_cache.pkl")
    args, remaining = parser.parse_known_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    selection = config["experiment"]
    target = Path(__file__).with_name(SCRIPTS[selection["type"]])
    listed = config_parts(config)
    if _given(remaining, "--part"):
        parts = [None]                                   # one part, chosen on the command line
    elif len(listed) > 1:
        parts = listed
    else:
        parts = [listed[0] if listed else None]
    if len(parts) > 1 and _given(remaining, "--out", "--cache"):
        parser.error("several parts: give --out_dir instead of --out / --cache")
    out_dir = args.out_dir or (os.path.join("results", Path(args.config).stem) if len(parts) > 1 else None)
    runs = []
    for part in parts:
        argv = list(remaining)
        if part is not None and len(parts) > 1:
            argv = ["--part", part, *argv]
        if out_dir is not None:
            if _given(remaining, "--out", "--cache"):
                parser.error("give either --out_dir or --out / --cache")
            name = part or next((t.split("=", 1)[1] if "=" in t else remaining[i + 1]
                                 for i, t in enumerate(remaining) if t.split("=", 1)[0] == "--part"), listed[0] if listed else "run")
            os.makedirs(out_dir, exist_ok=True)
            argv += ["--out", os.path.join(out_dir, f"{name}.json"), "--cache", os.path.join(out_dir, f"{name}_cache.pkl")]
        runs.append((part, argv))
    if len(runs) == 1:
        _run(target, args.config, runs[0][1])
        return
    for part, argv in runs:                              # check every part before running any
        try:
            _run(target, args.config, argv + ["--validate_only"])
        except SystemExit as exc:
            if exc.code not in (0, None):
                sys.exit(f"part {part}: the configuration is invalid (see above); nothing was run")
    failed = []
    for part, argv in runs:
        print(f"=== part {part}: {target.name} {' '.join(argv)}", flush=True)
        try:
            _run(target, args.config, argv)
        except SystemExit as exc:
            if exc.code not in (0, None):
                failed.append(part)
        except Exception:
            traceback.print_exc()
            failed.append(part)
    if failed:
        sys.exit(f"parts failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
