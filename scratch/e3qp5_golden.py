"""
The golden readings of tests/test_e0_field.py: the energies and force checksums of the rungs that existed before
v0.0.14 (M_inf, M_16pm, M_16C2pm, M_16C2pm_r1) at fixed perturbed parameters, on the test's three structures.  Run with
v0.0.13's model code (before any E_0-field change) and paste the printed dict into the test.

    python scratch/e3qp5_golden.py
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
spec = importlib.util.spec_from_file_location("tef", os.path.join(REPO, "tests", "test_e0_field.py"))
tef = importlib.util.module_from_spec(spec); spec.loader.exec_module(tef)
import e3_water as E3

structs = tef.structures(E3)
golden = {name: tef.readings(E3, tef.rung(E3, name), structs) for name in tef.BASE}
print("GOLDEN = {")
for name, rows in golden.items():
    print(f"    {name!r}: [" + ",\n               ".join("(" + ", ".join(repr(x) for x in row) + ")" for row in rows) + "],")
print("}")
