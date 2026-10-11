"""YAML experiment selection and water cluster dataset configuration."""
import math
from pathlib import Path
import yaml

TYPES = {"dimer_ladder", "water", "x2b_toy"}
FIELDS = {
    "dimer_ladder": {"systems", "truth", "rungs"},
    "water": {"part", "rungs"},
    "x2b_toy": {"orders"},
}
CLUSTER_DEFAULTS = {
    "train_sizes": "3,4,5", "test_sizes": "6,2",
    "n_train": 12, "n_test": 8, "train_seed": 0, "test_seed": 100,
    "mc_steps": 300, "temperature_kelvin": 300.0,
    "radius_angstrom": None, "min_oo_angstrom": 2.7,
    "min_atom_distance_angstrom": 1.7, "sigma": 1e-9,
    "monomer_references": 4,
}


def validate_cluster_data(data):
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    unknown = set(data) - set(CLUSTER_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported data fields: {', '.join(sorted(map(str, unknown)))}")
    for key, value in data.items():
        if key in ("train_sizes", "test_sizes"):
            if (not isinstance(value, list) or not value
                    or any(type(v) is not int or v < 2 for v in value)
                    or len(set(value)) != len(value)):
                raise ValueError(f"data.{key} must be a nonempty list of distinct integers >= 2")
        elif key in ("n_train", "n_test", "train_seed", "test_seed", "mc_steps", "monomer_references"):
            minimum = 1 if key in ("n_train", "n_test") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"data.{key} must be an integer >= {minimum}")
        else:
            if key == "radius_angstrom" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"data.{key} must be a finite number")
            if value < 0 or (key != "sigma" and value == 0):
                raise ValueError(f"data.{key} must be {'nonnegative' if key == 'sigma' else 'positive'}")

    resolved = {**CLUSTER_DEFAULTS, **data}
    radius = resolved["radius_angstrom"]
    if radius is not None and 2 * radius <= resolved["min_oo_angstrom"]:
        raise ValueError("data.radius_angstrom must exceed half of min_oo_angstrom to place multiple molecules")


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "experiment" not in config or set(config) - {"experiment", "data"}:
        raise ValueError("config must contain an experiment mapping and optional data mapping")
    selection = config["experiment"]
    if not isinstance(selection, dict):
        raise ValueError("experiment must be a mapping")
    kind = selection.get("type")
    if not isinstance(kind, str) or kind not in TYPES:
        raise ValueError("experiment.type must be dimer_ladder, water, or x2b_toy")
    unknown = set(selection) - FIELDS[kind] - {"type"}
    if unknown:
        raise ValueError(f"unsupported fields for {kind}: {', '.join(sorted(map(str, unknown)))}")
    for key in ("systems", "rungs", "orders"):
        if key not in selection:
            continue
        values = selection[key]
        valid = isinstance(values, list) and bool(values)
        if valid:
            valid = (all(type(v) is int and v in (0, 2) for v in values) if key == "orders"
                     else all(isinstance(v, str) and v and ',' not in v for v in values))
        if not valid:
            raise ValueError(f"experiment.{key} must be a nonempty list of {'orders 0 or 2' if key == 'orders' else 'names'}")
    if "truth" in selection and selection["truth"] not in ("tt", "published"):
        raise ValueError("experiment.truth must be tt or published")
    if "part" in selection and selection["part"] not in ("dimer", "clusters", "far", "ion"):
        raise ValueError("experiment.part must be dimer, clusters, far, or ion")
    if "data" in config:
        if kind != "water":
            raise ValueError("data configuration currently supports water clusters only")
        validate_cluster_data(config["data"])
    return config


def load_selection(path):
    return load_config(path)["experiment"]


def parse_selection(parser, kind, argv=None, *, rungs=None, systems=None):
    parser.add_argument("--config", help="YAML configuration; explicit CLI flags override it")
    probe, _ = parser.parse_known_args(argv)
    has_data = False
    if probe.config:
        try:
            config = load_config(probe.config)
            selection = config["experiment"]
            if selection["type"] != kind:
                raise ValueError(f"this script requires experiment.type: {kind}")
            defaults = {}
            for key, value in selection.items():
                if key == "type":
                    continue
                defaults["system" if key == "systems" else key] = ','.join(map(str, value)) if isinstance(value, list) else value
            has_data = "data" in config
            for key, value in config.get("data", {}).items():
                defaults[key] = ','.join(map(str, value)) if isinstance(value, list) else value
            parser.set_defaults(**defaults)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    if has_data and args.part != "clusters":
        parser.error("data configuration currently supports water part: clusters only")
    if kind == "water" and args.part == "clusters":
        try:
            data = {key: getattr(args, key, default) for key, default in CLUSTER_DEFAULTS.items()}
            for key in ("train_sizes", "test_sizes"):
                data[key] = [int(v) for v in data[key].split(',')]
            validate_cluster_data(data)
        except (ValueError, AttributeError) as exc:
            parser.error(str(exc))
    if rungs is not None:
        invalid = set(args.rungs.split(',')) - set(rungs)
        if invalid:
            parser.error(f"unknown rungs: {', '.join(sorted(invalid))}")
    if systems is not None:
        names = args.system.split(',')
        invalid = set(names) - set(systems[args.truth])
        if invalid:
            parser.error(f"unsupported systems for {args.truth}: {', '.join(sorted(invalid))}")
    if kind == "x2b_toy":
        if any(v not in ("0", "2") for v in args.orders.split(',')):
            parser.error("orders must be a comma-separated list of 0 or 2")
    return args
