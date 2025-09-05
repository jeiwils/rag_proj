import pandas as pd
import matplotlib.pyplot as plt

dense_retrieval_data = [
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "2wikimultihopqa", "Tokens/query": 1212.1, "Wall time/query (ms)": 5908.1, "TPS": 205.3, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 34.359, "Answer EM": 23.000, "APT (×10¹⁵ param tokens)": 1.697},
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "hotpotqa", "Tokens/query": 1316.9, "Wall time/query (ms)": 6439.4, "TPS": 204.7, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 46.871, "Answer EM": 35.000, "APT (×10¹⁵ param tokens)": 1.844},
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "musique", "Tokens/query": 2923.0, "Wall time/query (ms)": 6663.5, "TPS": 438.7, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 32.723, "Answer EM": 19.667, "APT (×10¹⁵ param tokens)": 4.092},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "2wikimultihopqa", "Tokens/query": 1217.5, "Wall time/query (ms)": 3252.2, "TPS": 374.6, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 17.912, "Answer EM": 4.000, "APT (×10¹⁵ param tokens)": 0.852},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "hotpotqa", "Tokens/query": 1348.1, "Wall time/query (ms)": 3814.2, "TPS": 353.8, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 23.262, "Answer EM": 9.333, "APT (×10¹⁵ param tokens)": 0.944},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "musique", "Tokens/query": 2974.0, "Wall time/query (ms)": 4151.2, "TPS": 717.0, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 16.804, "Answer EM": 5.000, "APT (×10¹⁵ param tokens)": 2.082},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "2wikimultihopqa", "Tokens/query": 763.9, "Wall time/query (ms)": 439.8, "TPS": 1750.2, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 19.469, "Answer EM": 5.667, "APT (×10¹⁵ param tokens)": 1.069},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "hotpotqa", "Tokens/query": 858.8, "Wall time/query (ms)": 458.8, "TPS": 1873.2, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 31.899, "Answer EM": 17.333, "APT (×10¹⁵ param tokens)": 1.202},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "musique", "Tokens/query": 2386.1, "Wall time/query (ms)": 742.6, "TPS": 3217.1, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 16.099, "Answer EM": 3.000, "APT (×10¹⁵ param tokens)": 3.341},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "2wikimultihopqa", "Tokens/query": 763.0, "Wall time/query (ms)": 745.7, "TPS": 1206.3, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 8.829, "Answer EM": 3.000, "APT (×10¹⁵ param tokens)": 1.068},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "hotpotqa", "Tokens/query": 863.4, "Wall time/query (ms)": 1101.8, "TPS": 958.9, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 10.298, "Answer EM": 3.333, "APT (×10¹⁵ param tokens)": 1.209},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "musique", "Tokens/query": 2386.6, "Wall time/query (ms)": 1069.6, "TPS": 2261.5, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 4.989, "Answer EM": 0.667, "APT (×10¹⁵ param tokens)": 3.341},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "2wikimultihopqa", "Tokens/query": 754.8, "Wall time/query (ms)": 160.5, "TPS": 4709.0, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 11.482, "Answer EM": 5.667, "APT (×10¹⁵ param tokens)": 0.528},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "hotpotqa", "Tokens/query": 852.0, "Wall time/query (ms)": 195.7, "TPS": 4384.3, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 30.892, "Answer EM": 18.333, "APT (×10¹⁵ param tokens)": 0.596},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "musique", "Tokens/query": 2379.7, "Wall time/query (ms)": 346.4, "TPS": 7018.7, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 8.497, "Answer EM": 1.667, "APT (×10¹⁵ param tokens)": 1.666},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "2wikimultihopqa", "Tokens/query": 755.3, "Wall time/query (ms)": 263.0, "TPS": 2875.4, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 31.020, "Answer EM": 22.333, "APT (×10¹⁵ param tokens)": 1.057},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "hotpotqa", "Tokens/query": 855.0, "Wall time/query (ms)": 372.7, "TPS": 2328.6, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 41.432, "Answer EM": 25.000, "APT (×10¹⁵ param tokens)": 1.197},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "musique", "Tokens/query": 2383.3, "Wall time/query (ms)": 651.0, "TPS": 3662.2, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.000, "Path EM": 0.000, "Answer F1": 23.049, "Answer EM": 6.333, "APT (×10¹⁵ param tokens)": 3.337},
]

combined_traversal_baseline_data = [
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "2wikimultihopqa", "Tokens/query": 12153.3, "Wall time/query (ms)": 9377.1, "TPS": 1301.1, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.012, "Path EM": 0.008, "Answer F1": 15.083, "Answer EM": 10.667, "APT (×10¹⁵ param tokens)": 17.015},
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "hotpotqa", "Tokens/query": 12229.8, "Wall time/query (ms)": 9856.4, "TPS": 1254.4, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.014, "Path EM": 0.016, "Answer F1": 34.523, "Answer EM": 26.333, "APT (×10¹⁵ param tokens)": 17.122},
    {"Model": "deepseek-r1-distill-qwen-14b", "Dataset": "musique", "Tokens/query": 12428.0, "Wall time/query (ms)": 11214.8, "TPS": 1124.5, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.018, "Path EM": 0.014, "Answer F1": 16.815, "Answer EM": 8.000, "APT (×10¹⁵ param tokens)": 17.399},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "2wikimultihopqa", "Tokens/query": 11801.3, "Wall time/query (ms)": 9245.3, "TPS": 1287.9, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.012, "Path EM": 0.008, "Answer F1": 16.373, "Answer EM": 12.000, "APT (×10¹⁵ param tokens)": 8.261},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "hotpotqa", "Tokens/query": 11704.5, "Wall time/query (ms)": 9062.3, "TPS": 1315.2, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.016, "Path EM": 0.016, "Answer F1": 33.849, "Answer EM": 26.333, "APT (×10¹⁵ param tokens)": 8.193},
    {"Model": "deepseek-r1-distill-qwen-7b", "Dataset": "musique", "Tokens/query": 11893.2, "Wall time/query (ms)": 10399.0, "TPS": 1151.9, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.019, "Path EM": 0.014, "Answer F1": 16.506, "Answer EM": 8.333, "APT (×10¹⁵ param tokens)": 8.325},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "2wikimultihopqa", "Tokens/query": 4793.5, "Wall time/query (ms)": 6904.1, "TPS": 706.0, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.026, "Path EM": 0.014, "Answer F1": 16.642, "Answer EM": 12.333, "APT (×10¹⁵ param tokens)": 6.711},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "hotpotqa", "Tokens/query": 5008.2, "Wall time/query (ms)": 6724.1, "TPS": 760.8, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.039, "Path EM": 0.030, "Answer F1": 33.988, "Answer EM": 25.000, "APT (×10¹⁵ param tokens)": 7.011},
    {"Model": "qwen2.5-14b-instruct", "Dataset": "musique", "Tokens/query": 4978.4, "Wall time/query (ms)": 9649.5, "TPS": 569.6, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.054, "Path EM": 0.028, "Answer F1": 16.416, "Answer EM": 8.000, "APT (×10¹⁵ param tokens)": 6.970},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "2wikimultihopqa", "Tokens/query": 7194.5, "Wall time/query (ms)": 8505.3, "TPS": 857.4, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.018, "Path EM": 0.010, "Answer F1": 16.363, "Answer EM": 12.333, "APT (×10¹⁵ param tokens)": 10.072},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "hotpotqa", "Tokens/query": 7190.7, "Wall time/query (ms)": 8557.8, "TPS": 858.1, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.024, "Path EM": 0.021, "Answer F1": 32.616, "Answer EM": 23.000, "APT (×10¹⁵ param tokens)": 10.067},
    {"Model": "qwen2.5-2x7b-moe-power-coder-v4", "Dataset": "musique", "Tokens/query": 7398.7, "Wall time/query (ms)": 9832.6, "TPS": 759.7, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.026, "Path EM": 0.018, "Answer F1": 19.076, "Answer EM": 9.667, "APT (×10¹⁵ param tokens)": 10.358},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "2wikimultihopqa", "Tokens/query": 5765.4, "Wall time/query (ms)": 6760.7, "TPS": 860.3, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.017, "Path EM": 0.012, "Answer F1": 13.438, "Answer EM": 9.667, "APT (×10¹⁵ param tokens)": 4.036},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "hotpotqa", "Tokens/query": 5663.0, "Wall time/query (ms)": 9613.8, "TPS": 671.4, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.034, "Path EM": 0.027, "Answer F1": 34.050, "Answer EM": 24.667, "APT (×10¹⁵ param tokens)": 3.964},
    {"Model": "qwen2.5-7b-instruct", "Dataset": "musique", "Tokens/query": 5816.6, "Wall time/query (ms)": 8364.2, "TPS": 706.1, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.042, "Path EM": 0.024, "Answer F1": 16.488, "Answer EM": 7.333, "APT (×10¹⁵ param tokens)": 4.072},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "2wikimultihopqa", "Tokens/query": 9526.5, "Wall time/query (ms)": 7281.2, "TPS": 1319.5, "Hit@k": 0.92, "Recall@k": 0.66, "Path F1": 0.014, "Path EM": 0.010, "Answer F1": 16.314, "Answer EM": 12.333, "APT (×10¹⁵ param tokens)": 13.337},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "hotpotqa", "Tokens/query": 9884.1, "Wall time/query (ms)": 7237.3, "TPS": 1389.3, "Hit@k": 1.00, "Recall@k": 0.92, "Path F1": 0.020, "Path EM": 0.018, "Answer F1": 32.456, "Answer EM": 24.667, "APT (×10¹⁵ param tokens)": 13.838},
    {"Model": "state-of-the-moe-rp-2x7b", "Dataset": "musique", "Tokens/query": 9518.3, "Wall time/query (ms)": 8374.8, "TPS": 1140.7, "Hit@k": 0.99, "Recall@k": 0.81, "Path F1": 0.026, "Path EM": 0.016, "Answer F1": 15.561, "Answer EM": 7.333, "APT (×10¹⁵ param tokens)": 13.326},
]

def make_table(data):
    cols = ["Model", "Dataset", "Tokens/query", "Wall time/query (ms)",
            "TPS", "Hit@k", "Recall@k", "Path F1", "Path EM", "Answer F1",
            "Answer EM", "APT (×10¹⁵ param tokens)"]
    return pd.DataFrame(data, columns=cols)

def save_table_as_jpg(df, filename):
    plt.figure(figsize=(14, len(df)*0.4 + 1))
    ax = plt.gca()
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dense_df = make_table(dense_retrieval_data)
    combined_df = make_table(combined_traversal_baseline_data)

    # Print to console
    print("Dense Retrieval\n")
    print(dense_df.to_string(index=False), "\n")
    print("Combined Traversal + Baseline Reading\n")
    print(combined_df.to_string(index=False))

    # Save tables as JPG
    save_table_as_jpg(dense_df, "dense_retrieval.jpg")
    save_table_as_jpg(combined_df, "combined_traversal_baseline.jpg")

if __name__ == "__main__":
    main()
