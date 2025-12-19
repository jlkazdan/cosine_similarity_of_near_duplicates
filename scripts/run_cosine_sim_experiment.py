import torch
from datasets import load_dataset
from src.measure_gradients import GradientComputer
from src.compute_similarities import compute_cosine_similarities
import random
import os

def main(model_name, dataset, num_examples, transformation, output_file):
    dataset = load_dataset(dataset, split="train", streaming=True)
    subset = list(dataset.take(num_examples))
    examples = [example["text"] for example in subset]


    #initalize the gradient computer
    gradient_computer = GradientComputer(model_name)

    if transformation == "shuffle":
        #first we need to get a baseline
        examples_shuffled = list(examples)
        random.shuffle(examples_shuffled)

        baseline = compute_cosine_similarities(examples, examples_shuffled, transformation = "none", gradient_computer = gradient_computer)
    
    else:
        perturbed = compute_cosine_similarities(examples, examples, transformation = transformation, gradient_computer = gradient_computer)

if __name__ == "__main__":
    model_name = "openai/gpt-oss-20b"
    dataset = "EleutherAI/fineweb-edu-dedup-10b"
    transformation = "translate_to_french"
    base_path = "/mnt/home/ServiceNowFundamentalResearch_scaling_of_memorization_gradients/data"
    output_name = "cosine_sims.csv"
    output_file = os.path.join(base_path, output_name)
    num_examples = 5
    main(model_name, dataset, num_examples, transformation, output_file)