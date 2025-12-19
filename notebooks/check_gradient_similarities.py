import numpy as np
import pandas as pd
import wandb


api = wandb.Api()

sweep = api.sweep("jkazdan/gradient-similarity-transformations/0dcmiwqf")
runs = sweep.runs

# Combine all run histories
import pandas as pd
all_metrics = []
for run in runs:
    metrics_df = run.history()
    metrics_df['run_id'] = run.id
    metrics_df['run_name'] = run.name
    all_metrics.append(metrics_df)

combined_df = pd.concat(all_metrics, ignore_index=True)
combined_df.to_csv("/mnt/home/ServiceNowFundamentalResearch_scaling_of_memorization_gradients/data/finished_sweep_data.csv")
