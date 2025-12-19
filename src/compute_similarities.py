import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from src.text_transformations import text_transformations
from src.measure_gradients import GradientComputer
from tqdm import tqdm
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def compute_memory_efficient_dot_product(grad1, grad2):
    """Compute cosine similarity between two gradients in a memory-efficient way."""
    norm1, norm2 = 0, 0
    dot = 0
    
    for ele1, ele2 in zip(grad1, grad2):
        norm1 += torch.norm(ele1).cpu()**2
        norm2 += torch.norm(ele2).cpu()**2
        dot += ele1.flatten().dot(ele2.flatten()).cpu()
    
    # Delete gradients immediately after use
    del grad1, grad2
    torch.cuda.empty_cache()
    
    return dot / torch.pow(norm1, 0.5) / torch.pow(norm2, 0.5)


def compute_cosine_similarities(
    corpus1: list[str], 
    corpus2: list[str], 
    transformation: str, 
    gradient_computer: GradientComputer, 
    output_file: str
):
    """
    Compute cosine similarities between gradients of two corpora.
    
    Args:
        corpus1: First list of text examples
        corpus2: Second list of text examples (will be transformed)
        transformation: Type of transformation to apply to corpus2
        gradient_computer: GradientComputer instance
        output_file: Path to save results
        
    Returns:
        List of cosine similarities
    """
    import hashlib
    import wandb
    
    def hash_text(text):
        """Create a short hash of text for logging."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    cosine_similarities = []
    
    # Write header if file doesn't exist
    import os
    write_header = not os.path.exists(output_file)
    
    with open(output_file, "a") as f:
        if write_header:
            f.write("model_name,example1,example2,transformation,cosine_similarity\n")
        
        # Add tqdm progress bar
        for i, (example1, example2) in enumerate(tqdm(
            zip(corpus1, corpus2), 
            total=len(corpus1),
            desc=f"Computing similarities ({transformation})",
            unit="example"
        )):
            # We cannot have more than 5000 characters in the translation APIs
            example1 = example1[:5000]
            example2 = example2[:5000]
            
            # Compute gradient for original example
            gradient_computer.compute_gradient(example1)
            original_gradient = gradient_computer.return_gradient()
            
            # Apply transformation and compute gradient
            transformed_example = text_transformations(example2, transformation)
            gradient_computer.compute_gradient(transformed_example)
            transformed_gradient = gradient_computer.return_gradient()
            
            # Compute cosine similarity
            cosine_sim = compute_memory_efficient_dot_product(original_gradient, transformed_gradient)
            # Note: gradients are deleted inside compute_memory_efficient_dot_product
            
            cosine_similarities.append(cosine_sim.item())
            
            # Explicitly delete the similarity tensor
            del cosine_sim
            
            # Log per-example similarity to wandb
            example1_hash = hash_text(example1)
            example2_hash = hash_text(example2)
            wandb.log({
                "example_similarity": cosine_similarities[-1],
                "example_index": i,
                "example1_hash": example1_hash,
                "example2_hash": example2_hash,
                "transformation": transformation
            })
            
            # Escape quotes and newlines in text for CSV
            example1_clean = example1.replace('"', '""').replace('\n', ' ')
            example2_clean = example2.replace('"', '""').replace('\n', ' ')
            
            # Write to CSV (with quotes around text fields)
            f.write(f'{gradient_computer.model_name},"{example1_clean}","{example2_clean}",{transformation},{cosine_similarities[-1]}\n')
            
            # Update tqdm postfix with current similarity
            tqdm.write(f"Example {i+1}: similarity = {cosine_similarities[-1]:.4f}")
            
            # Force garbage collection and cache clearing every iteration
            torch.cuda.empty_cache()
    
    return cosine_similarities