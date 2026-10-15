import glob, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import create_results as cr   # main() only runs under __main__, so this is safe

BASE = "data/voltage_control"
OUT = "compact_results/full_eval.csv"
cr.N_TEST = int(os.environ.get("N_TEST", 999999))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 0)) or None

def config_of(env_design):
    if "full_uniform" in env_design:
        return "uniform"
    if "normal_around_mean" in env_design:
        return "normal"
    return "time-series"

run_dirs = sorted(os.path.dirname(m) for m in
                  glob.glob(f"{BASE}/**/meta-data.txt", recursive=True))
if MAX_RUNS:
    run_dirs = run_dirs[:MAX_RUNS]

os.makedirs("compact_results", exist_ok=True)
rows, opt_objs, possible = [], None, None

for d in run_dirs:
    rel = os.path.relpath(d, BASE)
    if not os.path.exists(os.path.join(d, "actor.pth")):
        print("SKIP (no checkpoint):", rel)
        continue
    t0 = time.time()
    try:
        _, _, env_design = cr.get_algo_plus_hps(d, cr.REMOVE_HP)
        agent, env, _ = cr.get_agent_data(d)
        if opt_objs is None:
            opt_objs, possible = cr.get_opt_performance(env)
            print("Solvable test samples:", int(possible.sum()), "of", len(possible))
        objs, viol, rel_viol, valids = cr.get_performance(agent, env)
    except Exception as e:
        print("FAILED:", rel, repr(e))
        continue
    f = np.logical_and(valids, possible)
    if f.any():
        regret = cr.calculate_regret(objs[f], opt_objs[f])
        mpe_signed = cr.calculate_mpe(regret, opt_objs[f])
        mape = (np.abs(regret) / np.abs(opt_objs[f])).mean() * 100
        regret_mean = regret.mean()
    else:
        mpe_signed = mape = regret_mean = np.nan
    rows.append(dict(
        run=rel, config=config_of(env_design), n_test=len(valids),
        n_valid_solvable=int(f.sum()),
        mape=mape, mpe_signed=mpe_signed, regret_mean=regret_mean,
        invalid_pct=100 - cr.calculate_valid_share(valids, possible),
        viol_mean=np.mean(viol), rel_viol_mean=np.mean(rel_viol),
        seconds=round(time.time() - t0)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"done {rel}: MAPE={mape:.2f} invalid={rows[-1]['invalid_pct']:.2f}% ({rows[-1]['seconds']}s)")

df = pd.DataFrame(rows)
print()
print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
print("\nPer configuration:")
print(df.groupby("config").agg(
    n=("run", "count"),
    mape_mean=("mape", "mean"), mape_std=("mape", "std"), mape_median=("mape", "median"),
    mpe_signed_mean=("mpe_signed", "mean"),
    invalid_median=("invalid_pct", "median"),
).round(3).to_string())
