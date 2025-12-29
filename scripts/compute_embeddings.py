import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import wandb

from src.compute_embeddings import embed_all_and_save


if __name__ == "__main__":
    # Don't call wandb.agent here - the CLI does that
    # Just run the function directly for testing
    embed_all_and_save()