"""
The golden readings of tests/test_e0_field_cross.py: the energies and force checksums of E3-QP6's field rungs (M_16pm_w,
M_16C2pm_w, M_16C2pm_w2) at fixed perturbed parameters, on tests/test_e0_field.py's three structures, the field term's
taper as E3-QP6 ran it (5-6 A).  Run with v0.0.15's model code (before any v0.0.16 change) and paste the printed dict
into the test.

    python scratch/e3qp7_golden.py
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "tests"))
load = lambda name: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(name, os.path.join(REPO, "tests", name + ".py")))
tef, tew = load("test_e0_field"), load("test_e0_field_whole")
import e3_water as E3

structs = tef.structures(E3)
golden = {name: tef.readings(E3, tew.rung(E3, tef, name), structs) for name in tew.WHOLE}
print("GOLDEN_WHOLE = {")
for name, rows in golden.items():
    print(f"    {name!r}: [" + ",\n               ".join("(" + ", ".join(repr(x) for x in row) + ")" for row in rows) + "],")
print("}")
