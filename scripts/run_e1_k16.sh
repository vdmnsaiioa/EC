#!/bin/sh
# E1 / E2 at K = 16 on the published Ar2 and Ne2 potentials (phase-d-build-plan.md section 2).
# ~8 h on a 2-core CPU; minutes per rung on a GPU.  Outputs under results/.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
K=${K:-16}
for sys in Ar2 Ne2; do
  python3 scripts/dimer_ladder.py --system $sys --truth published --rungs M_inf,M_G,M_A,M_6,M_68,M_16p \
      --seeds $K --steps 1500 --lbfgs 300 --lA 4.0 --out results/e1_${sys}_published_K${K}.json \
      > results/e1_${sys}_published_K${K}.log 2>&1
done
python3 scripts/dimer_ladder.py --system Ne2,Ar2 --truth published --rungs M_6,M_68 --seeds $K --steps 1500 --lbfgs 300 \
    --out results/e2_NeAr_joint_K${K}_freeK.json > results/e2_NeAr_joint_K${K}_freeK.log 2>&1
python3 scripts/dimer_ladder.py --system Ne2,Ar2 --truth published --rungs M_6,M_68 --seeds $K --steps 1500 --lbfgs 300 \
    --one_oscillator --out results/e2_NeAr_joint_K${K}_oneosc.json > results/e2_NeAr_joint_K${K}_oneosc.log 2>&1
