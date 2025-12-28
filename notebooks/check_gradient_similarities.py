import numpy as np
import pandas as pd
import wandb
from notebooks.utils import get_model_params, compute_auc_vs_shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import os


REFRESH = False
SAVE_PATH = "/mnt/home/ServiceNowFundamentalResearch_scaling_of_memorization_gradients/notebooks/figures"



api = wandb.Api()

if REFRESH:
    sweep = api.sweep("jkazdan/gradient-similarity-transformations/0dcmiwqf")
    runs = sweep.runs

    # Combine all run histories
    all_metrics = []
    for run in runs:
        metrics_df = run.history(samples=10000)
        metrics_df['run_id'] = run.id
        metrics_df['run_name'] = run.name
        metrics_df['model'] = run.config.get('model_name')  # Access from config
        metrics_df['transformation'] = run.config.get('transformation')  # Access from config
        
        all_metrics.append(metrics_df)

    combined_df = pd.concat(all_metrics, ignore_index=True)
    combined_df.to_csv("/mnt/home/ServiceNowFundamentalResearch_scaling_of_memorization_gradients/data/finished_sweep_data.csv")


data = pd.read_csv("/mnt/home/ServiceNowFundamentalResearch_scaling_of_memorization_gradients/data/finished_sweep_data.csv")[['example1_hash', 'example2_hash', 'model', 'example_similarity', 'transformation']]
data["model_family"] = data["model"].apply(lambda x: x.split("/")[0])
data = data[data["example_similarity"] != 1.0]


data = data.reset_index(drop=True)
data['model_params'] = data['model'].apply(lambda x: get_model_params(x))
data["auc"] = data.groupby("model_params", group_keys=False).apply(compute_auc_vs_shuffle)
data["abs_example_similarity"] = np.abs(data["example_similarity"])
gb = data.groupby(by= ["transformation"])['example_similarity']


# similarity by transformation and num parameters
sns.lineplot(data = data, x = 'model_params', y = 'example_similarity', hue = 'transformation')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean Cosine Similarity")
name = "similarity_by_transformation_raw.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

# WITH MODEL FAMILY
sns.lineplot(data = data, x = 'model_params', y = 'example_similarity', hue = 'transformation', style='model_family')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean Cosine Similarity")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
name = "similarity_by_transformation_and_family_raw.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

# absolute simialrity by transfomration and num parameters
sns.lineplot(data = data, x = 'model_params', y = 'abs_example_similarity', hue = 'transformation')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean Cosine Similarity")
name = "abs_similarity_by_transformation_raw.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

# WITH MODEL FAMILY
sns.lineplot(data = data, x = 'model_params', y = 'abs_example_similarity', hue = 'transformation', style='model_family')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean Cosine Similarity")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
name = "abs_similarity_by_transformation_and_family_raw.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()


#now let's regularize by random transformations
shuffle_means = data[data["transformation"] == "shuffle"].groupby(by = "model")['example_similarity'].mean()
data["shuffle_mean"] = data["model"].map(shuffle_means)

shuffle_stds = data[data["transformation"] == "shuffle"].groupby(by = "model")['example_similarity'].std()
data["shuffle_std"] = data["model"].map(shuffle_stds)

data["z_score_rel_shuffled"] = (data["example_similarity"] - data["shuffle_mean"])/data["shuffle_std"]

sns.lineplot(data = data, x = 'model_params', y = "z_score_rel_shuffled", hue = "transformation")
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean z-score rel. baseline")
name = "z_score_rel_baseline_by_transformation.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

# WITH MODEL FAMILY
sns.lineplot(data = data, x = 'model_params', y = "z_score_rel_shuffled", hue = "transformation", style='model_family')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean z-score rel. baseline")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
name = "z_score_rel_baseline_by_transformation_and_family.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

sns.lineplot(data = data, x = 'model_params', y = "auc", hue = "transformation")
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean AUC rel. baseline")
name = "auc_by_transformation.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches = 'tight')
plt.close()

# WITH MODEL FAMILY (already exists)
sns.lineplot(data=data, x='model_params', y='auc', hue='transformation', style='model_family')
plt.xscale('log')
plt.xlabel("Model Parameters")
plt.ylabel("Mean AUC rel. baseline")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
name = "auc_by_transformation_and_family.png"
plt.savefig(os.path.join(SAVE_PATH, name), dpi=300, bbox_inches='tight')
plt.close()