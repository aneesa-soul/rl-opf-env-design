import glob, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import create_results as cr

PATTERNS = [
    "data/voltage_control/base_run1/*/",
    "data/voltage_control/uniform_run1/*/",
    "data/voltage_control/uniform_seeds/s2/*/",
    "data/voltage_control/uniform_seeds/s3/*/",
    "data/voltage_control/uniform_seeds/s4/*/",
    "data/voltage_control/uniform_seeds/s5/*/",
]
np.set_printoptions(precision=2, suppress=True)
for pat in PATTERNS:
    d = glob.glob(pat)[0]
    agent, env, _ = cr.get_agent_data(d)
    acts = []
    for step in list(env.unwrapped.test_steps)[:100]:
        obs, _ = env.reset(options={"step": step, "test": True})
        acts.append(np.asarray(agent.test_act(agent.scale_obs(obs))).ravel())
    acts = np.array(acts)
    print(pat.split("voltage_control/")[1])
    print("  mean action per actuator:", acts.mean(0))
    print("  std across 100 samples  :", acts.std(0))
