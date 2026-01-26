#!/usr/bin/env python3
import os, math, time, argparse
import numpy as np, faiss
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# Flare palette - sampled with maximum separation for 4 dimensions
# From dark red-brown through orange to pale yellow
COLORS = ["#2A0A18", "#A4133C", "#E85D04", "#FFCB69"]

def mmap(path, shape, mode): return np.memmap(path, np.float32, mode, shape=shape)

def materialize(big, big_shape, idx, d, out, chunk):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    need = not (os.path.exists(out) and os.path.getsize(out) == idx.size * d * 4)
    X = mmap(out, (idx.size, d), "w+" if need else "r+")
    if need:
        E = mmap(big, big_shape, "r")
        for i in tqdm(range(0, idx.size, chunk), desc="materialize"):
            j = idx[i:i+chunk]
            X[i:i+j.size] = E[j, :d]
        X.flush()
    return X

def normalize(X, chunk):
    for i in tqdm(range(0, X.shape[0], chunk), desc="normalize"):
        Y = X[i:i+chunk]
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

def pick_m(d):
    for m in (64,48,32,24,16,12,8):
        if d % m == 0: return m
    return d

# control when it finds the neighbors approximately versus exactly
def pick_nlist(n, train):
    # Need at least ~39 points per centroid for FAISS clustering
    # Fall back to exact search for small datasets
    if n < 2500:  # Below this, just use exact search
        return 0
    return max(64, min(int(math.sqrt(n)), train//2, n//50, 262144))

def build_index(X, nlist, m, train, nbits=8, add_chunk=200_000, gpu=None):
    n, d = X.shape
    if nlist == 0:
        ix = faiss.IndexFlatIP(d)
    else:
        q = faiss.IndexFlatIP(d)
        ts = min(train, n)
        nlist = min(int(nlist), ts)  # ensure train_points >= nlist
        ix = faiss.IndexIVFPQ(q, d, int(nlist), int(m), int(nbits), faiss.METRIC_INNER_PRODUCT)
        tr = np.asarray(X[np.random.default_rng(0).choice(n, ts, replace=False)], np.float32)
        t0 = time.time()
        ix.train(tr)
        print(f"train {time.time()-t0:.1f}s (nlist={nlist}, ts={ts})")
    if gpu is not None:
        ix = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), int(gpu), ix)
    for i in tqdm(range(0, n, add_chunk), desc="add"):
        ix.add(np.asarray(X[i:i+add_chunk], np.float32))
    return ix

def nn_cosines(ix, X, k=4, nprobe=16, qchunk=200_000, max_query=None):
    """
    Returns nearest-neighbor cosine similarity for each point, excluding self.
    Uses FAISS-returned inner products; X must already be L2-normalized.
    Clips to [-1,1] to enforce valid cosine range.
    """
    if hasattr(ix, "nprobe"): ix.nprobe = int(nprobe)
    n = X.shape[0]
    
    # FIX: determine how many queries we'll actually run
    largest = min(max_query, n) if max_query else n
    
    # FIX: only allocate what we need
    best = np.empty(largest, np.float32)

    for i in tqdm(range(0, largest, qchunk), desc="query"):
        e = min(i + qchunk, largest)  # FIX: bound by largest, not n
        sims, ids = ix.search(np.asarray(X[i:e], np.float32), int(k))  # sims: (B,k), ids: (B,k)

        self_ids = np.arange(i, e, dtype=np.int64)
        ok = ids != self_ids[:, None]          # exclude self
        j = np.argmax(ok, axis=1)             # first non-self (or 0 if none)
        best[i:e] = np.where(ok.any(1), sims[np.arange(e-i), j], -1.0)

    return np.clip(best, -1.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big_path", default="data/embeddings/embeddings_gemma300m.npy")
    ap.add_argument("--big_rows", type=int, default=190_168_005)
    ap.add_argument("--big_dim",  type=int, default=768)
    ap.add_argument("--amounts",  default="10,100,1000,3000,10000,30_000,100000,300_000,1000000,3_000_000,10000000,30_000_000,100000000")
    ap.add_argument("--dims",     default="768,512,256,128")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out_dir",  default="data/nn_results")
    ap.add_argument("--subset_dir", default="data/local_subsets")
    ap.add_argument("--fig_dir",  default="figures")
    ap.add_argument("--train",    type=int, default=500_000)
    ap.add_argument("--k",        type=int, default=4)
    ap.add_argument("--nprobe",   type=int, default=16)
    ap.add_argument("--nbits",    type=int, default=8)
    ap.add_argument("--gpu",      type=int, default=-1, help="GPU id, or -1 for CPU")
    ap.add_argument("--chunk",    type=int, default=200_000, help="materialize/add/query chunk")
    ap.add_argument("--query_number",    type=int, default=5_000_000, help="don't query the entire dataset")
    ap.add_argument("--norm_chunk", type=int, default=1_000_000)
    ap.add_argument("--fit_min", type=float, default=1e4, help="fit lines only to points with n >= this value")
    ap.add_argument("--fit_max", type=float, default=1e6, help="fit lines only to points with n <= this value")
    ap.add_argument("--hist_exclude", type=str, default="30000", help="comma-separated list of N values to exclude from histograms")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.subset_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    amounts = [int(x) for x in args.amounts.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    big_shape = (args.big_rows, args.big_dim)
    gpu = None if args.gpu < 0 else args.gpu

    results = {d: {"amounts": [], "means": [], "all": []} for d in dims}

    for n in amounts:
        idx = np.sort(rng.integers(0, args.big_rows, n, dtype=np.int64))
        for d in dims:
            out = f"{args.out_dir}/cosines_n{n}_d{d}.npy"
            if os.path.exists(out):
                print(f"load N={n:,} d={d}")
                cos = np.load(out)
                # FIX: only use valid entries (first min(query_number, n) values)
                valid_length = min(args.query_number, n)
                cos = cos[:valid_length]
            else:
                print(f"compute N={n:,} d={d}")
                X = materialize(args.big_path, big_shape, idx, d,
                                f"{args.subset_dir}/subset_n{n}_d{d}.f32", args.chunk)
                normalize(X, args.norm_chunk)
                ix = build_index(X, pick_nlist(n, args.train), pick_m(d),
                                 args.train, args.nbits, args.chunk, gpu)
                cos = nn_cosines(ix, X, args.k, args.nprobe, args.chunk, args.query_number)
                np.save(out, cos)

            results[d]["amounts"].append(n)
            results[d]["means"].append(float(cos.mean()))
            results[d]["all"].append(cos)

    np.savez(f"{args.out_dir}/summary_cosines.npz",
             data_amounts=np.array(amounts), dims=np.array(dims),
             **{f"means_d{d}": np.array(results[d]["means"], np.float32) for d in dims})

    # ---- Plotting: mean cosine (logx + loglog if you want) ----
    for log_y, name in [(False, "logx"), (True, "loglog")]:
        plt.figure(figsize=(10, 6))
        for i, d in enumerate(dims):
            color = COLORS[i % len(COLORS)]
            xs = np.array(results[d]["amounts"], dtype=np.float64)
            ys = -np.log(np.array(results[d]["means"], dtype=np.float64))
            
            # Only plot points >= fit_min
            plot_mask = xs >= args.fit_min
            plt.plot(xs[plot_mask], ys[plot_mask], "o-", color=color, label=f"d={d}", linewidth=2)
            
            # Fit line to points in [fit_min, fit_max]
            fit_mask = (xs >= args.fit_min) & (xs <= args.fit_max)
            if fit_mask.sum() >= 2:
                log_xs = np.log(xs[fit_mask])
                fit_ys = np.log(ys[fit_mask]) if log_y else ys[fit_mask]
                slope, intercept, _, _, _ = stats.linregress(log_xs, fit_ys)
                fit_x = np.logspace(np.log10(xs[plot_mask].min()), np.log10(xs[plot_mask].max()), 100)
                if log_y:
                    fit_y = np.exp(intercept + slope * np.log(fit_x))
                else:
                    fit_y = intercept + slope * np.log(fit_x)
                plt.plot(fit_x, fit_y, "--", color=color, alpha=0.6, linewidth=1.5)
        plt.xscale("log")
        if log_y: plt.yscale("log")
        plt.xlabel("Number of Data Points", fontsize=20)
        plt.ylabel("Average Nearest-Neighbor Cosine Similarity", fontsize=20)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{args.fig_dir}/nn_cosine_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---- Histograms of cosine ----
    # Only include amounts >= fit_min, excluding specified values
    hist_exclude = set(int(x) for x in args.hist_exclude.split(",") if x.strip())
    hist_amounts = [n for n in amounts if n >= args.fit_min and n not in hist_exclude]
    hist_indices = [i for i, n in enumerate(amounts) if n >= args.fit_min and n not in hist_exclude]
    
    num = len(hist_amounts)
    rows, cols = (2, 4) if num <= 8 else (int(math.ceil(num/4)), 4)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for plot_idx, (n, data_idx) in enumerate(zip(hist_amounts, hist_indices)):
        ax = axes[plot_idx]
        for i, d in enumerate(dims):
            color = COLORS[i % len(COLORS)]
            ax.hist(results[d]["all"][data_idx], bins=30, alpha=0.4, color=color, label=f"d={d}", density=True)
        ax.set_title(f"N={n:,}", fontsize=11)
        if plot_idx >= (rows-1)*cols: ax.set_xlabel("Cosine similarity", fontsize=20)
        if plot_idx % cols == 0: ax.set_ylabel("Density", fontsize=20)
        if plot_idx == 0: ax.legend(fontsize=9)

    for j in range(num, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{args.fig_dir}/nn_cosine_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Combined 2-panel figure: scaling law (left) + histograms (right) ----
    fig, (ax_scaling, ax_hist_panel) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: scaling law (loglog)
    for i, d in enumerate(dims):
        color = COLORS[i % len(COLORS)]
        xs = np.array(results[d]["amounts"], dtype=np.float64)
        ys = -np.log(np.array(results[d]["means"], dtype=np.float64))
        
        plot_mask = xs >= args.fit_min
        ax_scaling.plot(xs[plot_mask], ys[plot_mask], "o-", color=color, label=f"d={d}", linewidth=2)
        
        fit_mask = (xs >= args.fit_min) & (xs <= args.fit_max)
        if fit_mask.sum() >= 2:
            log_xs = np.log(xs[fit_mask])
            fit_ys = np.log(ys[fit_mask])
            slope, intercept, _, _, _ = stats.linregress(log_xs, fit_ys)
            fit_x = np.logspace(np.log10(xs[plot_mask].min()), np.log10(xs[plot_mask].max()), 100)
            fit_y = np.exp(intercept + slope * np.log(fit_x))
            ax_scaling.plot(fit_x, fit_y, "--", color=color, alpha=0.6, linewidth=1.5)
    
    ax_scaling.set_xscale("log")
    ax_scaling.set_yscale("log")
    ax_scaling.set_xlabel("Number of Data Points", fontsize=20)
    ax_scaling.set_ylabel(r"$-\log(\mathrm{mean\ NN\ cos\ sim})$", fontsize=20)
    ax_scaling.legend(fontsize=10)
    
    # Right panel: histograms as 2x2 subplots for specific N values
    ax_hist_panel.axis("off")
    
    # Specific N values for the 2x2 histogram grid
    combined_hist_ns = [100, 10_000, 1_000_000, 100_000_000]
    combined_hist_indices = []
    combined_hist_amounts = []
    for target_n in combined_hist_ns:
        for idx, n in enumerate(amounts):
            if n == target_n:
                combined_hist_indices.append(idx)
                combined_hist_amounts.append(n)
                break
    
    # Create 2x2 inset axes for histograms
    hist_rows, hist_cols = 2, 2
    bbox = ax_hist_panel.get_position()
    
    h_spacing = 0.03
    v_spacing = 0.1
    cell_width = (bbox.width - h_spacing * (hist_cols - 1)) / hist_cols
    cell_height = (bbox.height - v_spacing * (hist_rows - 1)) / hist_rows
    
    hist_axes = []
    for row in range(hist_rows):
        for col in range(hist_cols):
            left = bbox.x0 + col * (cell_width + h_spacing)
            bottom = bbox.y1 - (row + 1) * cell_height - row * v_spacing
            ax_inset = fig.add_axes([left, bottom, cell_width, cell_height])
            hist_axes.append(ax_inset)
    
    for plot_idx, (n, data_idx) in enumerate(zip(combined_hist_amounts, combined_hist_indices)):
        if plot_idx >= len(hist_axes):
            break
        ax = hist_axes[plot_idx]
        for i, d in enumerate(dims):
            color = COLORS[i % len(COLORS)]
            ax.hist(results[d]["all"][data_idx], bins=30, alpha=0.4, color=color, label=f"d={d}", density=True, range=(0, 1))
        ax.set_xlim(0, 1)
        ax.set_title(f"N={n:,}", fontsize=14)
        if plot_idx >= hist_cols:  # bottom row
            ax.set_xlabel("cos sim", fontsize=14)
        if plot_idx % hist_cols == 0:  # left column
            ax.set_ylabel("Density", fontsize=14)
        if plot_idx == 0:
            ax.legend(fontsize=9)
    
    plt.savefig(f"{args.fig_dir}/nn_cosine_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Fraction above cosine thresholds (3-panel figure) ----
    thresholds = [0.90, 0.95, 0.99]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, t in enumerate(thresholds):
        ax = axes[ax_idx]
        for i, d in enumerate(dims):
            color = COLORS[i % len(COLORS)]
            xs = np.array(results[d]["amounts"], dtype=np.float64)
            fracs = np.array([float(np.mean(cos > t)) for cos in results[d]["all"]])

            # Only plot points >= fit_min
            plot_mask = xs >= args.fit_min
            ax.plot(xs[plot_mask], fracs[plot_mask], "o-", color=color, label=f"d={d}", linewidth=4)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.7)
        ax.set_xlabel("Number of Data Points", fontsize=20)
        ax.set_ylabel(r"$\mathbb{P}(\mathrm{NN\ cos\ sim} > " + f"{t:.2f})$", fontsize=20)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:  # Only show legend on first panel
            ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{args.fig_dir}/nn_cosine_frac_gt_3panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- log(-log fraction) plots ----
    thresholds = [0.90, 0.95, 0.99]

    for t in thresholds:
        plt.figure(figsize=(10, 6))

        for i, d in enumerate(dims):
            color = COLORS[i % len(COLORS)]
            ns = np.array(results[d]["amounts"], dtype=np.float64)
            fracs = np.array([np.mean(cos > t) for cos in results[d]["all"]], dtype=np.float64)

            # Only plot points >= fit_min with valid fracs
            plot_mask = (ns >= args.fit_min) & (fracs > 0.0) & (fracs < 1.0)
            if not np.any(plot_mask):
                continue

            y = -np.log(fracs[plot_mask])
            plt.plot(
                ns[plot_mask],
                y,
                "o-",
                color=color,
                label=f"d={d}",
                linewidth=2,
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Data Points", fontsize=20)
        plt.ylabel(r"$-\log(\Pr(\mathrm{NN\ cos sim} > %.2f))$" % t, fontsize=20)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{args.fig_dir}/nn_cosine_neglog_frac_gt_{str(t).replace('.','p')}_loglog.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

if __name__ == "__main__":
    main()