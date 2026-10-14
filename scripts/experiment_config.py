"""YAML experiment selection, dataset settings, water model (architecture) and truth settings, multi-part water
configurations (experiment.part as a list, per-part overrides under `parts`), and shared training options."""
import argparse
import math
from pathlib import Path
import sys
import yaml

TYPES = {"dimer_ladder", "water", "x2b_toy"}
# The water model mapping: each YAML field maps to e3_water.py's existing CLI destination and a validation rule.
# band_edge_angstrom: null means the rung's own band edge (the CLI's --l1 0); the near-source switch radius of the
# band-field input is given as a multiple of the band edge (--rs_over_l1; 0 disables the switch).
MODEL_FIELDS = {
    "charges": ("charges", "charge_mode"),
    "charge_neutrality": ("charge_neutrality", "neutrality"),
    "band_edge_angstrom": ("l1", "optional_positive"),
    "rs_over_l1": ("rs_over_l1", "nonnegative"),
    "r_cut_angstrom": ("rcut", "positive"),
    "features": ("F", "positive_int"),
    "analytic_envelope_angstrom": ("lA", "positive"),
}
CHARGE_MODES = (None, "pinned", "learned")
NEUTRALITY_MODES = ("fragment", "structure")
FIELDS = {
    "dimer_ladder": {"systems", "truth", "rungs"},
    "water": {"part", "rungs"},
    "x2b_toy": {"orders"},
}
PARTS = ("dimer", "clusters", "far", "ion")
PART_SECTIONS = {"rungs", "model", "data", "training"}       # what a part may override under parts.<part>
TRUTH_NAMES = ("pw", "qaqua")
HELDOUT_NAMES = ("2b", "3b", "4b")
# The design of the water dimer part (its data mapping): the window [lower x R_e, ratio x that] with R_e the truth's own
# dimer minimum unless given, log-uniform nodes, the fixed orientation's acceptor tilt, label noise and its seed, the
# isolated monomers at zero, and the evaluation grid from 1.02 x the window's end to eval_max.
DIMER_DEFAULTS = {
    "re_angstrom": None, "window_lower_over_re": 1.35, "window_ratio": 1.8, "window_nodes": 40,
    "phi_degrees": 0.0, "sigma": 1e-9, "train_seed": 0, "monomer_references": 4,
    "eval_max_angstrom": 500.0, "eval_points": 61,
}
DIMER_RULES = {
    "re_angstrom": "optional_positive", "window_lower_over_re": "positive", "window_ratio": "positive",
    "window_nodes": "positive_int", "phi_degrees": "finite", "sigma": "nonnegative", "train_seed": "nonnegative_int",
    "monomer_references": "nonnegative_int", "eval_max_angstrom": "positive", "eval_points": "positive_int",
}
CLUSTER_DEFAULTS = {
    "train_sizes": "3,4,5", "test_sizes": "6,2",
    "n_train": 12, "n_test": 8, "train_seed": 0, "test_seed": 100,
    "mc_steps": 300, "temperature_kelvin": 300.0,
    "radius_angstrom": None, "min_oo_angstrom": 2.7,
    "min_atom_distance_angstrom": 1.7, "sigma": 1e-9,
    "monomer_references": 4,
}

# Each YAML field maps to the existing or shared CLI destination and a validation
# rule. Keeping this table shared makes YAML and CLI values follow the same rules.
TRAINING_FIELDS = {
    None: {
        "seeds": ("model_seeds", "seeds"),
        "energy_scale": ("energy_scale", "optional_positive"),
        "verbose": ("verbose", "bool"),
    },
    "adam": {
        "steps": ("steps", "nonnegative_int"),
        "learning_rate": ("learning_rate", "positive"),
        "final_learning_rate_ratio": ("final_learning_rate_ratio", "fraction"),
        "beta1": ("adam_beta1", "decay"),
        "beta2": ("adam_beta2", "decay"),
        "epsilon": ("adam_epsilon", "nonnegative"),
        "epsilon_root": ("adam_epsilon_root", "nonnegative"),
        "gradient_clip_norm": ("gradient_clip_norm", "optional_positive"),
        "batch_size": ("batch", "optional_positive_int"),
    },
    "readout": {
        "freeze_during_adam": ("freeze_readout", "bool"),
        "linear_solve": ("readout_linear_solve", "bool"),
        "ridge": ("readout_ridge", "nonnegative"),
    },
    "lbfgs": {
        "steps": ("lbfgs", "nonnegative_int"),
        "chunk_size": ("lbfgs_chunk", "positive_int"),
        "learning_rate": ("lbfgs_learning_rate", "optional_positive"),
        "memory_size": ("lbfgs_memory_size", "positive_int"),
        "scale_init_precond": ("lbfgs_scale_init_precond", "bool"),
        "max_linesearch_steps": ("lbfgs_max_linesearch_steps", "positive_int"),
        "readout_only": ("lbfgs_readout_only", "bool"),
    },
    "loss": {
        "energy_weight": ("w_energy", "nonnegative"),
        "force_weight": ("w_force", "nonnegative"),
        "far_weight": ("w_far", "positive"),
    },
}


def _validate_training_value(name, value, rule):
    if rule.startswith("optional_") and value is None:
        return
    rule = rule.removeprefix("optional_")
    if rule == "charge_mode":
        valid, description = value in CHARGE_MODES, "pinned, learned, or null"
    elif rule == "neutrality":
        valid, description = value in NEUTRALITY_MODES, "fragment or structure"
    elif rule == "seeds":
        valid = (isinstance(value, list) and bool(value)
                 and all(type(v) is int and v >= 0 for v in value)
                 and len(set(value)) == len(value))
        description = "a nonempty list of distinct nonnegative integers"
    elif rule == "bool":
        valid, description = type(value) is bool, "a boolean"
    elif rule in ("positive_int", "nonnegative_int"):
        minimum = 1 if rule == "positive_int" else 0
        valid = type(value) is int and value >= minimum
        description = f"an integer >= {minimum}"
    else:
        numeric = type(value) in (int, float) and math.isfinite(value)
        limits = {
            "positive": (lambda v: v > 0, "a finite positive number"),
            "nonnegative": (lambda v: v >= 0, "a finite nonnegative number"),
            "fraction": (lambda v: 0 <= v <= 1, "a finite number between 0 and 1"),
            "decay": (lambda v: 0 <= v < 1, "a finite number >= 0 and < 1"),
            "finite": (lambda v: True, "a finite number"),
        }
        condition, description = limits[rule]
        valid = numeric and condition(value)
    if not valid:
        raise ValueError(f"{name} must be {description}")


def validate_training(training):
    """Validate supplied YAML fields and return their CLI argument defaults."""
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    allowed = set(TRAINING_FIELDS[None]) | (set(TRAINING_FIELDS) - {None})
    unknown = set(training) - allowed
    if unknown:
        raise ValueError(f"unsupported training fields: {', '.join(sorted(map(str, unknown)))}")
    defaults = {}
    for section, fields in TRAINING_FIELDS.items():
        values = training if section is None else training.get(section, {})
        prefix = "training" if section is None else f"training.{section}"
        if not isinstance(values, dict):
            raise ValueError(f"{prefix} must be a mapping")
        if section is not None:
            unknown = set(values) - set(fields)
            if unknown:
                raise ValueError(f"unsupported {prefix} fields: {', '.join(sorted(map(str, unknown)))}")
        for key, (destination, rule) in fields.items():
            if key not in values:
                continue
            _validate_training_value(f"{prefix}.{key}", values[key], rule)
            defaults[destination] = (','.join(map(str, values[key]))
                                     if rule == "seeds" else values[key])
    return defaults


def _training_defaults(kind):
    return {
        "steps": 1000 if kind == "x2b_toy" else 1500,
        "lbfgs": 0 if kind == "dimer_ladder" else 300,
        "learning_rate": 3e-3, "final_learning_rate_ratio": 0.01,
        "adam_beta1": 0.9, "adam_beta2": 0.999,
        "adam_epsilon": 1e-8, "adam_epsilon_root": 0.0,
        "gradient_clip_norm": None,
        "batch": {"water": 12, "x2b_toy": 48, "dimer_ladder": None}[kind],
        "lbfgs_chunk": 6 if kind == "water" else 48,
        "freeze_readout": kind != "x2b_toy",
        "readout_linear_solve": kind != "x2b_toy", "readout_ridge": 1e-8,
        "lbfgs_learning_rate": None, "lbfgs_memory_size": 10,
        "lbfgs_scale_init_precond": True, "lbfgs_max_linesearch_steps": 20,
        "lbfgs_readout_only": False,
        "w_energy": 1.0, "w_force": 0.0 if kind == "x2b_toy" else 1.0,
        "w_far": 1.0, "energy_scale": None,
        "verbose": kind == "x2b_toy", "model_seeds": None,
    }


def add_training_arguments(parser, kind):
    """Add shared training flags after a script's existing legacy arguments."""
    defaults = _training_defaults(kind)
    parser.set_defaults(_training_kind=kind)
    parser._training_kind = kind
    integer_fields = {"steps", "lbfgs", "batch", "lbfgs_chunk",
                      "lbfgs_memory_size", "lbfgs_max_linesearch_steps"}
    boolean_fields = {"freeze_readout", "readout_linear_solve", "verbose",
                      "lbfgs_scale_init_precond", "lbfgs_readout_only"}
    help_text = {
        "steps": "Adam steps (readout frozen when --freeze_readout is enabled)",
        "lbfgs": "L-BFGS steps after Adam and the optional linear readout solve",
        "model_seeds": "comma-separated model initialization seeds; overrides YAML seeds",
        "freeze_readout": "freeze the field-response readout during Adam",
        "readout_linear_solve": "fit order-0 readout coefficients before L-BFGS",
        "lbfgs_readout_only": "train only the field-response readout during L-BFGS",
        "energy_scale": "positive target normalization; omitted means automatic",
    }
    for destination, default in defaults.items():
        option = f"--{destination}"
        if option in parser._option_string_actions:
            continue
        kwargs = {"default": default}
        if destination in boolean_fields:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif destination != "model_seeds":
            kwargs["type"] = int if destination in integer_fields else float
        if destination in help_text:
            kwargs["help"] = help_text[destination]
        parser.add_argument(option, **kwargs)
    seed_option = "--seed" if kind == "x2b_toy" else "--seeds"
    if seed_option not in parser._option_string_actions:
        parser.add_argument(seed_option, type=int, default=0 if kind == "x2b_toy" else 4)


def training_seeds(args):
    """Return the explicit seed list or the legacy seed/count selection."""
    selected = getattr(args, "model_seeds", None)
    if selected is not None:
        return [int(v) for v in selected.split(',')] if isinstance(selected, str) else list(selected)
    if hasattr(args, "seeds"):
        return list(range(args.seeds))
    return [getattr(args, "seed", 0)]


def training_kwargs(args):
    """Build fit keyword arguments, retaining defaults for direct legacy callers."""
    defaults = _training_defaults(getattr(args, "_training_kind", "water"))
    value = lambda key: getattr(args, key, defaults[key])
    return {
        "steps": value("steps"), "lr": value("learning_rate"),
        "w_energy": value("w_energy"), "w_force": value("w_force"),
        "batch_size": value("batch"), "lbfgs_steps": value("lbfgs"),
        "lbfgs_chunk": value("lbfgs_chunk"),
        "readout_adam": not value("freeze_readout"),
        "readout_linear_solve": value("readout_linear_solve"),
        "readout_ridge": value("readout_ridge"),
        "scale": value("energy_scale"), "verbose": value("verbose"),
        "lr_final_ratio": value("final_learning_rate_ratio"),
        "adam_b1": value("adam_beta1"), "adam_b2": value("adam_beta2"),
        "adam_eps": value("adam_epsilon"), "adam_eps_root": value("adam_epsilon_root"),
        "gradient_clip_norm": value("gradient_clip_norm"),
        "lbfgs_lr": value("lbfgs_learning_rate"),
        "lbfgs_memory": value("lbfgs_memory_size"),
        "lbfgs_scale_init_precond": value("lbfgs_scale_init_precond"),
        "lbfgs_max_linesearch_steps": value("lbfgs_max_linesearch_steps"),
        "lbfgs_readout_only": value("lbfgs_readout_only"),
    }


def _explicit_destinations(parser, argv):
    explicit = set()
    for token in argv:
        if token == "--":
            break
        option = token.split('=', 1)[0]
        action = parser._option_string_actions.get(option)
        if action is None and option.startswith('--') and parser.allow_abbrev:
            matches = [a for name, a in parser._option_string_actions.items() if name.startswith(option)]
            if matches and all(a is matches[0] for a in matches):
                action = matches[0]
        if action is not None:
            explicit.add(action.dest)
    return explicit


def _validate_resolved_training(args, kind, rungs):
    for section, fields in TRAINING_FIELDS.items():
        for key, (destination, rule) in fields.items():
            name = f"training.{key}" if section is None else f"training.{section}.{key}"
            value = training_seeds(args) if rule == "seeds" else getattr(args, destination)
            _validate_training_value(name, value, rule)
    if args.steps + args.lbfgs == 0:
        raise ValueError("training must include at least one Adam or L-BFGS step")
    if args.w_energy + args.w_force == 0:
        raise ValueError("training.loss must give positive weight to energy or force")
    if args.adam_epsilon + args.adam_epsilon_root == 0:
        raise ValueError("training.adam.epsilon and epsilon_root cannot both be zero")
    if kind == "x2b_toy":
        response = all_response = True
    elif isinstance(rungs, dict):
        response_models = [getattr(rungs[name], "band_fields", -1) >= 0
                           for name in args.rungs.split(',')]
        response, all_response = any(response_models), all(response_models)
    else:
        # A minimal parser may not have the model definitions. In that case
        # defer architecture-specific validation to fit rather than guessing.
        response = all_response = None
    if response and args.freeze_readout and args.lbfgs == 0:
        raise ValueError("a frozen field-response readout needs training.lbfgs.steps > 0")
    if args.lbfgs_readout_only:
        if all_response is False:
            raise ValueError("training.lbfgs.readout_only requires every selected model to have field response")
        if args.lbfgs == 0:
            raise ValueError("training.lbfgs.readout_only requires training.lbfgs.steps > 0")


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


def validate_dimer_data(data):
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    unknown = set(data) - set(DIMER_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported dimer data fields: {', '.join(sorted(map(str, unknown)))}")
    for key, value in data.items():
        _validate_training_value(f"data.{key}", value, DIMER_RULES[key])
    resolved = {**DIMER_DEFAULTS, **data}
    if resolved["window_ratio"] <= 1:
        raise ValueError("data.window_ratio must exceed 1")
    if resolved["window_nodes"] < 2 or resolved["eval_points"] < 2:
        raise ValueError("data.window_nodes and data.eval_points must be at least 2")


def validate_data(part, data):
    """the data mapping of a water part: the dimer design or the cluster data sets."""
    if part == "dimer":
        validate_dimer_data(data)
    elif part == "clusters":
        validate_cluster_data(data)
    else:
        raise ValueError(f"data configuration supports the water parts dimer and clusters, not {part}")


def validate_truth(truth, kind):
    """the water truth mapping -- name (pw or qaqua), root (the q-AQUA build directory), heldout (the raw CCSD(T)
    n-body sets read in the clusters part: sets, rigid_tolerance and strict_tolerance as {oh_angstrom,
    angle_degrees}, max_pair_energy_kcal (the training domain: every pair's truth 2-body energy at most this; null:
    no cap), max_frames) -- returned as the CLI argument defaults it sets."""
    if kind != "water":
        raise ValueError("truth configuration supports water only (the dimer ladder selects its truth with experiment.truth)")
    if not isinstance(truth, dict):
        raise ValueError("truth must be a mapping")
    unknown = set(truth) - {"name", "root", "heldout"}
    if unknown:
        raise ValueError(f"unsupported truth fields: {', '.join(sorted(map(str, unknown)))}")
    defaults = {}
    if "name" in truth:
        if truth["name"] not in TRUTH_NAMES:
            raise ValueError("truth.name must be pw or qaqua")
        defaults["truth"] = truth["name"]
    if "root" in truth:
        if not isinstance(truth["root"], str) or not truth["root"]:
            raise ValueError("truth.root must be a path")
        defaults["qaqua_root"] = truth["root"]
    if "heldout" in truth:
        held = truth["heldout"]
        if held is None:
            defaults["heldout"] = "none"
        else:
            if not isinstance(held, dict):
                raise ValueError("truth.heldout must be a mapping or null")
            unknown = set(held) - {"sets", "rigid_tolerance", "strict_tolerance", "max_pair_energy_kcal", "max_frames"}
            if unknown:
                raise ValueError(f"unsupported truth.heldout fields: {', '.join(sorted(map(str, unknown)))}")
            sets = held.get("sets") or []
            if (not isinstance(sets, list) or any(v not in HELDOUT_NAMES for v in sets) or len(set(sets)) != len(sets)):
                raise ValueError("truth.heldout.sets must be a list of distinct names from 2b, 3b, 4b")
            defaults["heldout"] = ",".join(sets) if sets else "none"
            for name, dests in (("rigid_tolerance", ("heldout_oh_tol", "heldout_angle_tol")),
                                ("strict_tolerance", ("heldout_strict_oh_tol", "heldout_strict_angle_tol"))):
                if name not in held:
                    continue
                tol = held[name]
                if not isinstance(tol, dict) or set(tol) - {"oh_angstrom", "angle_degrees"}:
                    raise ValueError(f"truth.heldout.{name} must be a mapping of oh_angstrom and angle_degrees")
                for key, dest in zip(("oh_angstrom", "angle_degrees"), dests):
                    if key in tol:
                        _validate_training_value(f"truth.heldout.{name}.{key}", tol[key], "positive")
                        defaults[dest] = tol[key]
            if "max_pair_energy_kcal" in held:
                _validate_training_value("truth.heldout.max_pair_energy_kcal", held["max_pair_energy_kcal"], "optional_positive")
                defaults["heldout_max_pair_kcal"] = held["max_pair_energy_kcal"] or 0.0
            if "max_frames" in held:
                _validate_training_value("truth.heldout.max_frames", held["max_frames"], "optional_positive_int")
                defaults["heldout_max"] = held["max_frames"] or 0
    return defaults


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def config_parts(config):
    """the water parts a config lists: experiment.part as a string or a list."""
    part = config["experiment"].get("part")
    return list(part) if isinstance(part, list) else ([part] if part else [])


def part_config(config, part):
    """one part of a multi-part configuration: the shared sections with the part's own mapping under parts.<part>
    merged over them (nested mappings key by key, everything else replaced) and experiment.part set to it."""
    own = (config.get("parts") or {}).get(part) or {}
    out = {key: value for key, value in config.items() if key != "parts"}
    out["experiment"] = {**config["experiment"], "part": part}
    if "rungs" in own:
        out["experiment"]["rungs"] = own["rungs"]
    for section in ("model", "data", "training"):
        if section in own:
            out[section] = _merge(config.get(section) or {}, own[section])
    return out


def validate_model(model, kind):
    """Validate the water model mapping (charge source and neutrality, band edge, near-source switch, E_0's cutoff
    and width, M_A's envelope) and return the CLI argument defaults it sets; omitted fields keep the script's
    defaults, charges: null keeps each rung's own charge source."""
    if kind != "water":
        raise ValueError("model configuration currently supports water only")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    unknown = set(model) - set(MODEL_FIELDS)
    if unknown:
        raise ValueError(f"unsupported model fields: {', '.join(sorted(map(str, unknown)))}")
    defaults = {}
    for key, (destination, rule) in MODEL_FIELDS.items():
        if key not in model:
            continue
        _validate_training_value(f"model.{key}", model[key], rule)
        value = model[key]
        if key == "band_edge_angstrom" and value is None:
            value = 0.0                                   # --l1 0: the rung's own band edge
        defaults[destination] = value
    return defaults


def _validate_resolved_model(args):
    """The architecture as resolved from YAML and CLI (a minimal parser may lack some of the flags)."""
    for key, (destination, rule) in MODEL_FIELDS.items():
        if not hasattr(args, destination):
            continue
        value = getattr(args, destination)
        if key == "band_edge_angstrom":
            value = None if value == 0 else value
        _validate_training_value(f"model.{key}", value, rule)


def _validate_sections(config, kind, part):
    """the selection lists and the data / model / training / truth mappings of one (part of a) configuration."""
    selection = config["experiment"]
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
    if "data" in config:
        if kind != "water":
            raise ValueError("data configuration supports water only")
        validate_data(part or "clusters", config["data"])
    if "model" in config:
        validate_model(config["model"], kind)
    if "training" in config:
        validate_training(config["training"])
    if "truth" in config:
        validate_truth(config["truth"], kind)


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if (not isinstance(config, dict) or "experiment" not in config
            or set(config) - {"experiment", "data", "training", "model", "truth", "parts"}):
        raise ValueError("config must contain an experiment mapping and optional data/training/model/truth/parts mappings")
    selection = config["experiment"]
    if not isinstance(selection, dict):
        raise ValueError("experiment must be a mapping")
    kind = selection.get("type")
    if not isinstance(kind, str) or kind not in TYPES:
        raise ValueError("experiment.type must be dimer_ladder, water, or x2b_toy")
    unknown = set(selection) - FIELDS[kind] - {"type"}
    if unknown:
        raise ValueError(f"unsupported fields for {kind}: {', '.join(sorted(map(str, unknown)))}")
    if "truth" in selection and selection["truth"] not in ("tt", "published"):
        raise ValueError("experiment.truth must be tt or published")
    if "part" in selection:
        part = selection["part"]
        if isinstance(part, list):
            if not part or any(p not in PARTS for p in part) or len(set(part)) != len(part):
                raise ValueError("experiment.part must be dimer, clusters, far, or ion, or a list of distinct parts")
        elif part not in PARTS:
            raise ValueError("experiment.part must be dimer, clusters, far, or ion")
    parts = config.get("parts")
    if parts is not None:
        if kind != "water":
            raise ValueError("parts are supported for water only")
        if not isinstance(parts, dict) or not parts:
            raise ValueError("parts must be a nonempty mapping of part names to mappings")
        for name, own in parts.items():
            if name not in PARTS:
                raise ValueError(f"unknown part {name!r} under parts (choose from {', '.join(PARTS)})")
            if not isinstance(own, dict):
                raise ValueError(f"parts.{name} must be a mapping")
            unknown = set(own) - PART_SECTIONS
            if unknown:
                raise ValueError(f"unsupported fields under parts.{name}: {', '.join(sorted(map(str, unknown)))}")
        if "data" in config:
            raise ValueError("with parts, give each part's data mapping under parts.<part>.data")
        for name in config_parts(config) + [p for p in parts if p not in config_parts(config)]:
            try:
                _validate_sections(part_config(config, name), kind, name)
            except ValueError as exc:
                raise ValueError(f"part {name}: {exc}") from None
    else:
        listed = config_parts(config)
        _validate_sections(config, kind, listed[0] if len(listed) == 1 else None)
    return config


def load_selection(path):
    return load_config(path)["experiment"]


def parse_selection(parser, kind, argv=None, *, rungs=None, systems=None, validate=None):
    """parse the script's arguments with a YAML configuration's values as defaults (explicit flags override them).
    A multi-part water configuration resolves the part from --part (or its single listed part) and merges that
    part's overrides.  validate(args) adds the script's own checks; --validate_only exits after every check."""
    parser.add_argument("--config", help="YAML configuration; explicit CLI flags override it")
    parser.add_argument("--validate_only", action="store_true",
                        help="check the configuration and the resolved arguments, then exit without running")
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = _explicit_destinations(parser, argv)
    probe, _ = parser.parse_known_args(argv)
    data = None
    training_defaults = {}
    if probe.config:
        try:
            config = load_config(probe.config)
            if config["experiment"]["type"] != kind:
                raise ValueError(f"this script requires experiment.type: {kind}")
            listed = config_parts(config)
            if "part" in explicit:
                part = probe.part
            elif len(listed) > 1:
                raise ValueError(f"this configuration lists the parts {', '.join(listed)}: select one with --part, "
                                 "or run them in sequence with scripts/run_experiment.py")
            else:
                part = listed[0] if listed else None
            effective = part_config(config, part) if config.get("parts") and part is not None else config
            defaults = {}
            for key, value in effective["experiment"].items():
                if key == "type" or (key == "part" and isinstance(value, list)):
                    continue
                defaults["system" if key == "systems" else key] = ','.join(map(str, value)) if isinstance(value, list) else value
            data = effective.get("data")
            for key, value in (data or {}).items():
                defaults[key] = ','.join(map(str, value)) if isinstance(value, list) else value
            training_defaults = validate_training(effective.get("training", {}))
            defaults.update(training_defaults)
            if "model" in effective:
                defaults.update(validate_model(effective["model"], kind))
            if "truth" in effective:
                defaults.update(validate_truth(effective["truth"], kind))
            parser.set_defaults(**defaults)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    if data is not None:
        try:
            validate_data(args.part, data)
        except ValueError as exc:
            parser.error(f"data configuration for part {args.part}: {exc}")
    if kind == "water" and args.part == "clusters":
        try:
            resolved = {key: getattr(args, key, default) for key, default in CLUSTER_DEFAULTS.items()}
            for key in ("train_sizes", "test_sizes"):
                resolved[key] = [int(v) for v in resolved[key].split(',')]
            validate_cluster_data(resolved)
        except (ValueError, AttributeError) as exc:
            parser.error(str(exc))
    if kind == "water" and args.part == "dimer":
        try:
            validate_dimer_data({key: getattr(args, key, default) for key, default in DIMER_DEFAULTS.items()})
        except ValueError as exc:
            parser.error(str(exc))
    if kind == "water":
        try:
            _validate_resolved_model(args)
        except (ValueError, TypeError) as exc:
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
    if hasattr(parser, "_training_kind"):
        legacy_seed = "seed" if kind == "x2b_toy" else "seeds"
        if legacy_seed in explicit:
            if "model_seeds" in explicit:
                parser.error(f"--model_seeds cannot be combined with --{legacy_seed}")
            args.model_seeds = None
        # Historically the water dimer workflow uses full-batch Adam and a
        # larger L-BFGS chunk; explicit YAML/CLI choices override that default.
        if kind == "water" and args.part == "dimer":
            for destination, default in (("batch", None), ("lbfgs_chunk", 48)):
                if destination not in training_defaults and destination not in explicit:
                    setattr(args, destination, default)
        try:
            _validate_resolved_training(args, kind, rungs)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
    if validate is not None:
        try:
            validate(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.validate_only:
        print(f"configuration valid: {kind}" + (f", part {args.part}" if hasattr(args, "part") else ""), flush=True)
        sys.exit(0)
    return args
