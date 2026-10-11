"""Dispatch a YAML-selected experiment; remaining flags go to its script."""
import argparse
import runpy
import sys
from pathlib import Path
import yaml
from experiment_config import load_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", required=True)
    args, remaining = parser.parse_known_args()
    try:
        selection = load_selection(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    scripts = {"dimer_ladder": "dimer_ladder.py", "water": "e3_water.py", "x2b_toy": "x2b_toy_regression.py"}
    target = Path(__file__).with_name(scripts[selection["type"]])
    sys.argv = [str(target), "--config", args.config, *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
