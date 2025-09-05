import matplotlib.pyplot as plt

# averaged metrics from repository
points = [
    (1316.88, 46.87, "deepseek-r1-distill-qwen-14b"),
    (251.26, 33.99,  "qwen2.5-14b-instruct"),
    (254.97, 32.62,  "qwen2.5-2x7B MoE"),
    (251.53, 32.46,  "state-of-the-moe-rp-2x7B"),
]

x, y, labels = zip(*points)
plt.scatter(x, y)
for xi, yi, lbl in points:
    plt.annotate(lbl, (xi, yi))
plt.xlabel("Tokens per query")
plt.ylabel("Answer F1 (mean)")
plt.title("HotpotQA: accuracy vs. token usage")
plt.show()
import matplotlib.pyplot as plt

models = [
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-2×7B-MoE-PowerCoder-v4",
    "State-of-the-MoE-RP-2×7B",
]

# Averages across 3 datasets × 3 seeds (above table)
f1_scores  = [37.98, 19.33, 22.49, 16.96, 8.04, 31.83]
wall_times = [6.337, 3.739, 0.547, 0.234, 0.972, 0.429]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(wall_times, f1_scores, s=120, alpha=0.7)

for model, x, y in zip(models, wall_times, f1_scores):
    ax.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("Inference Wall Time / Query (s)")
ax.set_ylabel("Answer Quality (F1)")
ax.set_title("Answer Quality vs Inference Wall Time\n(average over 3 seeds)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
