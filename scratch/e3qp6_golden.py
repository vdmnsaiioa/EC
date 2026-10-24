"""
The golden readings of tests/test_e0_field_whole.py: the energies and force checksums of E3-QP5's field rungs (M_16pm_f,
M_16C2pm_f, M_inf_f) at fixed perturbed parameters, on tests/test_e0_field.py's three structures.  Run with v0.0.14's
model code (before any v0.0.15 change) and paste the printed dict into the test.

    python scratch/e3qp6_golden.py
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
spec = importlib.util.spec_from_file_location("tef", os.path.join(REPO, "tests", "test_e0_field.py"))
tef = importlib.util.module_from_spec(spec); spec.loader.exec_module(tef)
import e3_water as E3

structs = tef.structures(E3)
golden = {name: tef.readings(E3, tef.rung(E3, name), structs) for name in tef.FIELD}
print("GOLDEN_FIELD = {")
for name, rows in golden.items():
    print(f"    {name!r}: [" + ",\n               ".join("(" + ", ".join(repr(x) for x in row) + ")" for row in rows) + "],")
print("}")
