"""
The far-field-matched response of E3-QP2, computed before any training: for each read-out rung of
configs/water_qaqua_pol2.yaml, the read-out (species coefficients, independent of E_0) solved on the training probes
that E_0 cannot see -- their interaction energies, exactly as T.fit(readout_first=True) does -- and read on the test
probes and the test hexamers.  Beyond r_c the trained models' probe readings are these numbers (E_0 cancels in the
probe's interaction and the read-out is held), so they are the pre-registered values.

    python scratch/e3qp2_matched_response.py [build_pol] [rungs] [--quick]
"""
import os, sys, time, json
import numpy as np
import jax

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from sseft import water as W, model as M, linear as L, train as T, structure as S
from sseft.units import ang_to_bohr, bohr_to_ang
import e3_water as E3

K = 627.5094740631
args = [x for x in sys.argv[1:] if not x.startswith("--")]
edge = float(next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--edge=")), 0.0))
root = args[0] if args else os.path.join(REPO, "external", "qaqua", "build_pol")
names = (args[1] if len(args) > 1 else "M_16S0pm,M_16C0pm,M_16S2pm,M_16C2pm,M_16C0p").split(",")
quick = "--quick" in sys.argv
truth = W.qaqua_pol_truth(root)
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
t0 = time.time()

# the data of configs/water_qaqua_pol2.yaml's clusters part
geo = dict(T_K=300.0, radius_ang=None, min_OO=2.7, min_any=1.7)
dists = (8.0, 10.0, 12.0, 15.0)
n_cores, per = (4, 1) if quick else (16, 2)
train_probes = []
for k, n_core in enumerate((3, 4, 5)):
    made = W.probe_configurations(n_core, dists, per, seed=400 + 1000 * k, mc_steps=300, n_cores=n_cores, truth=truth, **geo)[0]
    for st in made:
        st.info.update(n_core=n_core, core_id=1000 * k + st.info["core"])
    train_probes += made
test_probes = W.probe_configurations(3, dists, 10, seed=500, mc_steps=300, n_cores=2, truth=truth, **geo)[0]
hexamers = W.cluster_dataset([6], 24, seed=100, truth=truth, mc_steps=300, **geo)
print(f"{len(train_probes)} training probes, {len(test_probes)} test probes, {len(hexamers)} hexamers  [{time.time() - t0:.0f} s]", flush=True)

a = type("A", (), dict(lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0, response_edge=edge, rungs=",".join(names)))()
blind = E3.far_field_blind(train_probes, a.rcut)
labelled = E3.probe_interaction_labels(train_probes, truth)
match = [labelled[i] for i in np.flatnonzero(blind)]
groups = np.array([st.info["core_id"] for st in match])
ref = E3.probe_truth(test_probes, truth)
test_blind = E3.far_field_blind(test_probes, a.rcut)
print(f"matching on {len(match)} of {len(train_probes)} training probes (beyond r_c = 6 A of their cores: "
      + ", ".join(f"{d:g} A {np.mean(blind[np.array([s.info['d'] for s in train_probes]) == d]) * 100:.0f} %" for d in dists)
      + f"); test probes beyond r_c: {test_blind.mean() * 100:.0f} %", flush=True)
for key, order in (("first", 1), ("second", 2)):
    e = ref[f"nonadditive_induction_{key}"] - ref["nonadditive"]
    print(f"  the truth's non-additive energy by TTM3-F's induction through order {order} in alpha: residual "
          + ", ".join(f"{d:g} A {100 * rms(e[ref['d'] == d]) / rms(ref['nonadditive'][ref['d'] == d]):.1f} %" for d in dists), flush=True)
hex_ind = np.array([truth.qa.ttm(s.positions)["ind"] - sum(
    truth.qa.ttm(np.concatenate([s.positions[3 * i:3 * i + 3], s.positions[3 * j:3 * j + 3]]))["ind"]
    for i in range(6) for j in range(i + 1, 6)) for s in hexamers])
print(f"  the test hexamers' many-body induction (TTM3-F) rms {rms(hex_ind) * K:.3f} kcal/mol", flush=True)

# the raw 3-body set (near-rigid, in the training domain) at largest O-O 7-9 and >= 9 A, with the PES's own error
import math
from sseft import qaqua as QA
pos3, lab3 = QA.read_raw(os.path.join(truth.qa.root, "q-AQUA", "3b_data.xyz"), 3)
keep = np.flatnonzero(QA.near_rigid(pos3, W.R_OH, math.degrees(W.THETA), 0.03, 5.0))
roo = QA.largest_oo(pos3[keep])
raw_bins = []
for lo, hi in ((7.0, 9.0), (9.0, math.inf)):
    sel = keep[(roo >= lo) & (roo < hi)]
    sel = np.array([f for f in sel if max(truth.energy(np.concatenate([pos3[f][3 * i:3 * i + 3], pos3[f][3 * j:3 * j + 3]]), 2)
                                       for i in range(3) for j in range(i + 1, 3)) * K <= 2.0])
    P = pos3[sel]; Lb = lab3[sel]
    blind3 = np.array([min(np.linalg.norm(p[3 * i:3 * i + 3, None] - p[None, 3 * j:3 * j + 3], axis=-1).min()
                           for i in range(3) for j in range(i + 1, 3)) > a.rcut for p in P])
    pes = np.array([truth.qa.nbody(ang_to_bohr(p), max_order=3)[3] for p in P])
    print(f"  raw 3-body {lo:g}-{hi:g} A: {len(P)} frames ({blind3.sum()} with every pair beyond r_c), labels rms {rms(Lb) * K:.4f}, "
          f"the PES's rmse {100 * rms(pes - Lb) / rms(Lb):.0f} % ({100 * rms(pes[blind3] - Lb[blind3]) / rms(Lb[blind3]):.0f} % beyond r_c)", flush=True)
    raw_bins.append(((lo, hi), P, Lb, blind3))

out = {}
for name in names:
    t1 = time.time()
    rung = E3.configure(E3.RUNGS[name], a)
    structs = [s for s in match]
    pins = lambda xs: E3.pin_for(rung, xs, truth)
    pins(structs); pins(test_probes); pins(hexamers)
    params = M.init_params(jax.random.PRNGKey(0), rung, energy_scale=1e-3)
    # E_0 switched off (its site-energy head's output layer zero): every reading below is the pinned bands and the
    # matched response alone -- what the trained models read wherever E_0 is blind (the solve does not depend on it)
    last = params["heads"]["energy"][-1]
    params["heads"]["energy"][-1] = {"w": 0 * last["w"], "b": 0 * last["b"]}
    data = S.pad_batch(structs, n_freq=rung.K)
    if rung.band_fields < 0:                                   # the daggers: the pinned bands alone
        rep = L.SolveReport([(0.0, 0.0)], [0.0])
    else:
      params, rep = L.solve_readout(params, rung, data, 1e-3, w_force=0.0, w_energy=1.0, chunk=64, orders="all",
                                    iterations=4 if rung.bf_mutual else 1, ridge_grid=L.RIDGE_GRID, groups=groups)
    model = E3.probe_model(params, rung, test_probes, truth, a)
    m = E3.probe_metrics(model, ref)
    line = E3.probe_summary(m)
    eight = (ref["d"] == 8.0) & test_blind
    m["blind_8"] = {"count": int(eight.sum()), "nonadditive": rms(model[1][eight] - ref["nonadditive"][eight]) / rms(ref["nonadditive"][eight])}
    line += f"; 8 A beyond r_c ({eight.sum()}) {100 * m['blind_8']['nonadditive']:.0f} %"
    # the raw CCSD(T) trimers with largest O-O 7-9 and >= 9 A: the response's 3-body energy against the labels (E_0 blind
    # when every intermolecular atom pair is beyond r_c)
    rows = []
    for (lo, hi), P, Lb, blind3 in raw_bins:
        e3 = E3.model_nbody(params, rung, P, np.array([truth.molecule_charges(p) for p in P]), 3, truth)
        rows.append(f"{lo:g}-{hi:g} A {100 * rms(e3 - Lb) / rms(Lb):.0f} % (beyond r_c {blind3.sum()}: "
                    f"{100 * rms(e3[blind3] - Lb[blind3]) / rms(Lb[blind3]):.0f} %)")
        m[f"raw3b_{lo:g}"] = rms(e3 - Lb) / rms(Lb)
    line += "\n    raw 3-body: " + "; ".join(rows)
    # the response's own many-body induction on the hexamers: its energy minus its 15 dimers' (the read-out alone)
    if rung.band_fields < 0:
        print(f"{name}: test probes {line}", flush=True)
        out[name] = {"probes": m}
        continue
    aux = T.predict_aux(params, rung, hexamers)
    pairs = [W.water_structure(bohr_to_ang(np.concatenate([s.positions[3 * i:3 * i + 3], s.positions[3 * j:3 * j + 3]])),
                               charges=truth.molecule_charges(bohr_to_ang(np.concatenate([s.positions[3 * i:3 * i + 3], s.positions[3 * j:3 * j + 3]]))))
             for s in hexamers for i in range(6) for j in range(i + 1, 6)]
    pins(pairs)
    aux_p = T.predict_aux(params, rung, pairs)
    hex_resp = np.asarray(aux["E_bf"]) - np.asarray(aux_p["E_bf"]).reshape(len(hexamers), 15).sum(1)
    # the response's whole energy in the clusters (what E_0 trains next to) against TTM3-F's induction: hexamers, dimers
    hex_tot = np.array([truth.qa.ttm(s.positions)["ind"] for s in hexamers])
    dimers = [W.water_structure(bohr_to_ang(s.positions), charges=truth.molecule_charges(bohr_to_ang(s.positions)))
              for s in W.cluster_dataset([2], 24, seed=200, truth=truth, mc_steps=300, **geo)]
    pins(dimers)
    dim_resp = np.asarray(T.predict_aux(params, rung, dimers)["E_bf"]); dim_ind = np.array([truth.qa.ttm(s.positions)["ind"] for s in dimers])
    print(f"{name}: solve {', '.join(f'{b:.3e} -> {c:.3e}' for b, c in rep.history)} (ridge {', '.join(f'{r:.0e}' for r in rep.ridge)}); "
          f"[{time.time() - t1:.0f} s]\n    test probes {line}\n    hexamers: the response's many-body energy against TTM3-F's many-body "
          f"induction: residual {100 * rms(hex_resp - hex_ind) / rms(hex_ind):.0f} % (its rms {rms(hex_resp) * K:.3f} kcal/mol); its whole "
          f"energy {np.mean(np.asarray(aux['E_bf'])) * K:+.2f} kcal/mol on average against TTM3-F's induction {np.mean(hex_tot) * K:+.2f} "
          f"(rmse {rms(np.asarray(aux['E_bf']) - hex_tot) * K:.2f}); test dimers {np.mean(dim_resp) * K:+.3f} against {np.mean(dim_ind) * K:+.3f} "
          f"(rmse {rms(dim_resp - dim_ind) * K:.3f})", flush=True)
    rec = {"probes": m, "solve": rep.history, "ridge": rep.ridge, "hexamer_response_residual": rms(hex_resp - hex_ind) / rms(hex_ind)}
    if rung.bf_mutual:
        for k in (1, 2):
            mk = E3.probe_metrics(E3.probe_model(params, M.with_rung(rung, bf_mutual=k), test_probes, truth, a), ref)
            rec[f"order{k}"] = mk
            print(f"    its read-out through order {k} in the polarisabilities: {E3.probe_summary(mk)}", flush=True)
    rec.update(hexamer_whole_rmse=rms(np.asarray(aux["E_bf"]) - hex_tot), dimer_rmse=rms(dim_resp - dim_ind))
    out[name] = rec
    import pickle
    pickle.dump(jax.tree_util.tree_map(np.asarray, params["readout"]), open(os.path.join(REPO, "scratch", f"matched_{name}{'_quick' if quick else ''}.pkl"), "wb"))
tag = ("_quick" if quick else "") + (f"_edge{edge:g}" if edge else "")
json.dump(out, open(os.path.join(REPO, "scratch", f"e3qp2_matched_response{tag}.json"), "w"), indent=1, default=float)
print(f"done in {time.time() - t0:.0f} s")
