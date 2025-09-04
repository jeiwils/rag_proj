"""


Create a violin plot for a per-query metric averaged across seeds.
"""

import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory holding seed subdirectories (e.g. baseline_seed1, baseline_seed2, baseline_seed3)
base = Path("data/results/qwen2.5-14b-instruct/musique/dev")
metric = "reader_total_tokens"  # or reader_prompt_tokens, t_reader_ms, etc.

rows = []
for seed_dir in sorted(base.glob("*seed*")):
    with open(seed_dir / "token_usage.json") as f:
        per_query = json.load(f)["per_query_reader"]
    for qid, values in per_query.items():
        rows.append({"query": qid, "seed": seed_dir.name, metric: values[metric]})

# Build a DataFrame and average across seeds for each query
df = pd.DataFrame(rows)
mean_df = df.groupby("query")[metric].mean().reset_index()

# Create the violin plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.violinplot(y=metric, data=mean_df, inner="quartile")
plt.title(f"{metric} per query (mean across seeds)")
plt.xlabel("")
plt.tight_layout()
plt.show()
# To save instead of show, replace the last line with:
# plt.savefig("violin_plot.png")
