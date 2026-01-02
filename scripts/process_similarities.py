import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# DATA_PATH = 'data/embeddings/embeddings_gemma300m.npy'
# EWMA_ALPHA = 0.8
def ewma(points, alpha):
    assert alpha>0
    assert alpha<=1
    avg = points[0]
    ret_points = [points[0]]

    for point in points[1:]:
        val = ret_points[-1]*alpha + point * (1-alpha)
        ret_points.append(val)
    return ret_points

# # Memory-mapped array (read-only)
# embds = np.memmap(DATA_PATH, dtype=np.float32, mode='r', shape=(190168005, 768))

# # Get indices to check
indices = np.logspace(1, 7, 100).astype(int)#7.301029995663981, 200).astype(int)
# print(max(indices))
# similarities = []
# for idx in indices:
#     # Load and normalize only the current row
#     row = embds[idx].copy()  # copy to get actual array
#     row = row / np.linalg.norm(row)
    
#     # Load and normalize previous rows in chunks to avoid memory issues
#     chunk_size = 10000
#     max_sim = -1
    
#     for start in range(0, idx, chunk_size):
#         end = min(start + chunk_size, idx)
#         chunk = embds[start:end].copy()
#         # Normalize each row in chunk
#         norms = np.linalg.norm(chunk, axis=1, keepdims=True)
#         chunk = chunk / norms
#         # Compute similarities
#         sims = chunk @ row
#         max_sim = max(max_sim, np.max(sims))
    
#     similarities.append(max_sim)
#     print(f"Index {idx}: max similarity = {max_sim:.4f}")

# print(similarities)
similarities = [0.23804098, 0.3174304, 0.23137116, 0.4518083, 0.4043405, 0.260171, 0.26394978, 0.34112543, 0.39614818, 0.2909093, 0.34727818, 0.21308444, 0.28537402, 0.4200032, 0.3098327, 0.25527957, 0.37840122, 0.3639791, 0.35220385, 0.46249247, 0.35061365, 0.38819462, 0.42659274, 0.3191965, 0.48620245, 0.6630794, 0.41839853, 0.33980632, 0.55086434, 0.45610327, 0.4835148, 0.58128893, 0.36285526, 0.36402354, 0.41623265, 0.36606574, 0.35593143, 0.43963706, 0.5142463, 0.6592764, 0.48220614, 0.39273524, 0.3977209, 0.44202533, 0.4961758, 0.4711852, 0.5611752, 0.65105414, 0.52660817, 0.5964405, 0.52487546, 0.5789542, 0.6558773, 0.46508867, 0.83360606, 0.74812704, 0.8308772, 0.80652297, 0.8010725, 0.6034833, 0.6486996, 0.5471128, 0.68710554, 0.68702257, 0.5741254, 0.7959701, 0.618613, 0.8089069, 0.6876159, 0.619174, 0.64895564, 0.7006834, 0.5002949, 0.6540327, 0.78253245, 0.6588997, 0.66601455, 0.71737075, 0.73782086, 0.6800957, 0.553154, 0.81502783, 0.6818019, 0.7582075, 0.85540926, 0.7591866, 0.64005554, 0.67223614, 0.6335503, 0.6927786, 0.826594, 0.80299425, 0.78557885, 0.8374178, 0.770342, 0.7425477, 0.8143922, 0.80704767, 0.7145362, 0.9012886]
ewma_sims = ewma(similarities, 0.8)
data = {"similarities": similarities, "index": indices, "smoothed": ewma_sims}
data = pd.DataFrame(data)
data["neg_log_similarities"] = -np.log(data["similarities"])
data["neg_log_smoothed_similarities"] = -np.log(data["smoothed"])

# Create figures folder if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Then plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='index', y='neg_log_similarities', label='Raw', alpha=0.5)
sns.lineplot(data=data, x='index', y='neg_log_smoothed_similarities', label='Smoothed (EWMA)')
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Max Similarity')
plt.title('Embedding Similarities vs Training Position')
plt.legend()
plt.savefig('figures/embedding_similarities.png', dpi=300, bbox_inches='tight')
plt.show()
