import glob, os, re
import pandas as pd

def field(text, key):
    m = re.search(rf"^{key}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""

rows = []
for m in sorted(glob.glob("data/**/meta-data.txt", recursive=True)):
    d = os.path.dirname(m)
    t = open(m).read()
    envh = field(t, "Environment hyperparams")
    if "full_uniform" in envh:
        cfg = "uniform"
    elif "normal_around_mean" in envh:
        cfg = "normal"
    elif envh == "{}":
        cfg = "time-series"
    else:
        cfg = "other"
    target = int(field(t, "Training steps") or 0)
    csv = os.path.join(d, "test_returns.csv")
    reached = int(pd.read_csv(csv)["step"].iloc[-1]) if os.path.exists(csv) else 0
    rows.append(dict(
        path=os.path.relpath(d, "data"),
        config=cfg,
        seed=field(t, "Seed"),
        reached=reached,
        target=target,
        finished=(reached >= target > 0),
    ))

df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 90)
print(df.to_string(index=False))
print("\nFinished runs per configuration:")
print(df[df.finished].groupby("config").size().to_string())
print("\nDuplicate seeds (should be none):", df[df.duplicated("seed", keep=False)].shape[0])
