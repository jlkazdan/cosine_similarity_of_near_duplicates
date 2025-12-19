import torch
import numpy as np
from datasets import load_dataset
from src.measure_gradients import GradientComputer
from src.compute_similarities import compute_cosine_similarities
import random
import os
from pathlib import Path
import wandb

def main():
    # Initialize wandb
    run = wandb.init()
    
    # Get hyperparameters from wandb config
    config = wandb.config
    
    model_name = config.model_name
    dataset_name = config.dataset
    transformation = config.transformation
    num_examples = config.num_examples
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    subset = list(dataset.take(num_examples))
    examples = [example["text"] for example in subset]

    # Initialize the gradient computer
    gradient_computer = GradientComputer(model_name)

    # Set up output file path
    script_dir = Path(__file__).parent.resolve()
    base_path = script_dir / "data"
    base_path.mkdir(parents=True, exist_ok=True)
    output_name = f"cosine_sims_{model_name.replace('/', '_')}_{transformation}.csv"
    output_file = base_path / output_name

    # Compute similarities based on transformation type
    if transformation == "shuffle":
        # First we need to get a baseline
        examples_shuffled = list(examples)
        random.shuffle(examples_shuffled)
        
        similarities = compute_cosine_similarities(
            examples, 
            examples_shuffled, 
            transformation="none", 
            gradient_computer=gradient_computer,
            output_file=str(output_file)
        )
    else:
        similarities = compute_cosine_similarities(
            examples, 
            examples, 
            transformation=transformation, 
            gradient_computer=gradient_computer,
            output_file=str(output_file)
        )
    
    # Log results to wandb
    if similarities is not None:
        # Assuming similarities is a numpy array or list of similarities
        mean_similarity = float(torch.tensor(similarities).mean()) if torch.is_tensor(similarities) else float(np.mean(similarities))
        std_similarity = float(torch.tensor(similarities).std()) if torch.is_tensor(similarities) else float(np.std(similarities))
        
        wandb.log({
            "mean_cosine_similarity": mean_similarity,
            "std_cosine_similarity": std_similarity,
            "min_cosine_similarity": float(torch.tensor(similarities).min()) if torch.is_tensor(similarities) else float(np.min(similarities)),
            "max_cosine_similarity": float(torch.tensor(similarities).max()) if torch.is_tensor(similarities) else float(np.max(similarities)),
        })
        
    wandb.finish()

if __name__ == "__main__":
    main()