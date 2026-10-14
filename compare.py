import glob, os, re
import pandas as pd

BASE = "data/voltage_control"
NAMES = (("base", "time-series"), ("uniform", "uniform"), ("normal", "normal"))
TS = re.compile(r"/\d{4}-\d\d-\d\dT[\d.]+_QMarketEnv_Ddpg1Step_")

def config_of(path):
    p = path.replace("\\", "/")
    for key, name in NAMES:
        if f"voltage_control/{key}_" in p:
            return name
    return None

rows = []
for f in sorted(glob.glob(f"{BASE}/**/test_returns.csv", recursive=True)):
    cfg = config_of(f)
    if cfg is None:
        continue
    df = pd.read_csv(f)
    tail = df.tail(3)
    rel = os.path.relpath(os.path.dirname(f), BASE)
    rows.append(dict(
        config=cfg,
        run=TS.sub("/", rel),
        step=int(df["step"].iloc[-1]),
        valid_mape_final=df["valid_mape"].iloc[-1],
        valid_mape_last3=tail["valid_mape"].mean(),
        mape_all_last3=tail["mape"].mean(),
        invalid_pct_last3=100 * (1 - tail["valid_share"].mean()),
    ))

runs = pd.DataFrame(rows).sort_values(["config", "run"])
print(runs.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

done = runs[runs.step >= 1_000_000]
print("\nFinished runs only (1M steps), last-3-evaluation averages:")
summary = done.groupby("config").agg(
    n=("run", "count"),
    valid_mape_median=("valid_mape_last3", "median"),
    valid_mape_mean=("valid_mape_last3", "mean"),
    valid_mape_std=("valid_mape_last3", "std"),
    mape_all_median=("mape_all_last3", "median"),
    invalid_pct_median=("invalid_pct_last3", "median"),
)
print(summary.round(2).to_string())

os.makedirs("compact_results", exist_ok=True)
runs.to_csv("compact_results/my_runs.csv", index=False)
