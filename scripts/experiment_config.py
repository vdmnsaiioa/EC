"""Selection-only YAML configuration shared by experiment entry points."""
from pathlib import Path
import yaml

TYPES = {"dimer_ladder", "water", "x2b_toy"}
FIELDS = {
    "dimer_ladder": {"systems", "truth", "rungs"},
    "water": {"part", "rungs"},
    "x2b_toy": {"orders"},
}


def load_selection(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"experiment"}:
        raise ValueError("config must contain only an experiment mapping")
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
            valid = all(type(v) is int and v in (0, 2) for v in values) if key == "orders" else all(isinstance(v, str) and v and ',' not in v for v in values)
        if not valid:
            raise ValueError(f"experiment.{key} must be a nonempty list of {'orders 0 or 2' if key == 'orders' else 'names'}")
    if "truth" in selection and selection["truth"] not in ("tt", "published"):
        raise ValueError("experiment.truth must be tt or published")
    if "part" in selection and selection["part"] not in ("dimer", "clusters", "far", "ion"):
        raise ValueError("experiment.part must be dimer, clusters, far, or ion")
    return selection


def parse_selection(parser, kind, argv=None, *, rungs=None, systems=None):
    parser.add_argument("--config", help="YAML experiment selection; explicit CLI flags override it")
    probe, _ = parser.parse_known_args(argv)
    if probe.config:
        try:
            selection = load_selection(probe.config)
            if selection["type"] != kind:
                raise ValueError(f"this script requires experiment.type: {kind}")
            defaults = {}
            for key, value in selection.items():
                if key == "type":
                    continue
                defaults["system" if key == "systems" else key] = ','.join(map(str, value)) if isinstance(value, list) else value
            parser.set_defaults(**defaults)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
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
