# plot_timeline.py
import pandas as pd
import matplotlib.pyplot as plt

# Edit these to match your actual file names from main.py
final_csv = "logs/final_run_A_dense.csv"
base_csv  = "logs/baseline_run_A_dense.csv"

final_df = pd.read_csv(final_csv)
base_df  = pd.read_csv(base_csv)

plt.figure(figsize=(9, 3))
plt.step(base_df["time_s"], base_df["state_id"], where="post", label="Baseline (reflex)")
plt.step(final_df["time_s"], final_df["state_id"], where="post", label="Final (model-based + heuristics)")
plt.yticks([0,1,2,3], ["IDLE","ON_BREAK","STUDYING","DISTRACTED"])
plt.xlabel("Time (s)")
plt.ylabel("Agent state")
plt.title("FSM State Over Time (1‑minute trial)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure3_fsm_state_timeline.png", dpi=300, bbox_inches="tight")
print("Saved Figure 3 to figure3_fsm_state_timeline.png")
