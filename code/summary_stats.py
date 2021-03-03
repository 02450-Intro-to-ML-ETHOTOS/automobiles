import os
from load_data import * # also loads numpy and pandas

# make output dir
os.makedirs("../out/stats", exist_ok=True)

# summary statistics for numerical values
sumstats_numerical_attrs = raw_data.describe()
sumstats_numerical_attrs.to_csv("../out/stats/sumstats_numerical_attrs.tsv", sep="\t")

# summmary statistics for categorical values
categorical_attrs = [c for c in raw_data.columns.values if c not in sumstats_numerical_attrs.columns.values]
sumstats_categorical_attrs = {a: raw_data[a].value_counts().to_frame() for a in categorical_attrs}

# calculate pct for each categorical attribute and save result
N = raw_data.shape[0]
for a, stats in sumstats_categorical_attrs.items():
    sumstats_categorical_attrs[a].rename(columns={a: "count"}, inplace=True)
    sumstats_categorical_attrs[a]["pct"] = stats / N
    stats.to_csv(f"../out/stats/sumstats_categorical_attrs_{a}.tsv", sep="\t")
