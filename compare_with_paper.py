import json
import numpy as np
import pandas as pd
from scipy import stats

KEYS = {
    "time-series": "qmarket_{}_{}",
    "normal": "qmarket_{'train_data': 'normal_around_mean', 'sampling_kwargs': {'std': 0.3}}_{}",
    "uniform": "qmarket_{'train_data': 'full_uniform'}_{}",
}
ours = pd.read_csv("compact_results/full_eval.csv")
paper = {m: json.load(open(f"compact_results/{m}.json")) for m in ("mpe", "invalid")}

def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.maximum(np.array([np.var(g, ddof=1) for g in groups]), 1e-12)
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    t = ((1 - w / W) ** 2 / (n - 1)).sum()
    F = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * t)
    df2 = (k ** 2 - 1) / (3 * t)
    return F, stats.f.sf(F, k - 1, df2)

def summ(x):
    x = np.asarray(x, float)
    return f"n={len(x)} mean={x.mean():.2f} median={np.median(x):.2f} min={x.min():.2f} max={x.max():.2f}"

for metric, col, jname in (("MAPE (%)", "mape", "mpe"), ("Invalid share (%)", "invalid_pct", "invalid")):
    print("\n==", metric, "==")
    for cfg, key in KEYS.items():
        o = ours.loc[ours.config == cfg, col].values
        p = np.array(paper[jname][key], float)
        mw = stats.mannwhitneyu(o, p, alternative="two-sided").pvalue
        wt = stats.ttest_ind(o, p, equal_var=False).pvalue
        print(f"{cfg:12s} ours : {summ(o)}")
        print(f"{'':12s} paper: {summ(p)}")
        print(f"{'':12s} yours vs paper: Mann-Whitney p={mw:.3f}, Welch t p={wt:.3f}")
    groups = [ours.loc[ours.config == c, col].values for c in KEYS]
    F, p = welch_anova(groups)
    print(f"Welch ANOVA across your 3 groups: F={F:.1f}, p={p:.2e}")
