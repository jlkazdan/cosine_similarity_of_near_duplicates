import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import wandb
import os
import json

def embed_all_and_save(config=None):
    # Initialize wandb
    with wandb.init(config=config):
        config = wandb.config
        
        print(f"\n{'='*60}")
        print(f"Starting: {config.model_name}")
        print(f"Output: {config.output_file}")
        print(f"{'='*60}\n")

        os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
        
        # Checkpoint file to track progress
        checkpoint_file = config.output_file + ".checkpoint"
        
        # Load model in bfloat16
        print("Loading model in bfloat16...")
        model = SentenceTransformer(
            config.model_name, 
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        model.eval()
        model.max_seq_length = config.max_seq_length
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        # Start multi-process pool
        if num_gpus > 1:
            pool = model.start_multi_process_pool()
            print(f"Started multi-GPU pool with {num_gpus} GPUs")
        
        # Load dataset
        if config.dataset_name == "fineweb-edu-dedup":
            corpus_full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
                split="train",
                num_proc=min(16, os.cpu_count()),
            )

        texts = corpus_full_dataset["text"]
        print(f"Loaded {len(texts)} texts")
        
        # Warmup and get embedding dim
        print("Warming up...")
        sample_texts = texts[:256]
        if num_gpus > 1:
            sample = model.encode_multi_process(sample_texts, pool, batch_size=128, show_progress_bar=True)
        else:
            sample = model.encode(sample_texts, batch_size=128, show_progress_bar=True)
        
        print(sample[0])
        # Convert to float32 for consistency
        sample = sample.astype(np.float32)
        embedding_dim = sample.shape[1]
        
        # Check if checkpoint exists and determine starting point
        start_idx = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_idx = checkpoint_data['last_completed_idx']
                print(f"Found checkpoint: resuming from index {start_idx}")
        
        wandb.config.update({
            "embedding_dim": embedding_dim, 
            "num_texts": len(texts),
            "num_gpus": num_gpus,
            "dtype": "bfloat16",
            "resume_from": start_idx
        })
        
        # Create or open memory-mapped file
        print(f"{'Resuming' if start_idx > 0 else 'Creating'} output file: {config.output_file}")
        mode = 'r+' if start_idx > 0 else 'w+'
        all_embds = np.memmap(config.output_file, dtype=np.float32, mode=mode, 
                              shape=(len(texts), embedding_dim))
        
        if start_idx >= len(texts):
            print("All texts already embedded. Exiting.")
            if num_gpus > 1:
                model.stop_multi_process_pool(pool)
            return
        
        t0 = time.time()
        
        # Process in chunks starting from start_idx
        num_chunks = (len(texts) - start_idx + config.save_every - 1) // config.save_every
        pbar = tqdm(initial=start_idx, total=len(texts), desc="Embedding texts", unit="texts")
        
        for i in range(start_idx, len(texts), config.save_every):
            end_idx = min(i + config.save_every, len(texts))
            chunk_texts = texts[i:end_idx]
            
            # Use multi-GPU encoding if available
            print(f"The number of gpus is: {num_gpus}")
            if num_gpus > 1:
                chunk_embds = model.encode_multi_process(
                    chunk_texts, 
                    pool,
                    batch_size=config.batch_size, 
                    show_progress_bar=True
                )
            else:
                chunk_embds = model.encode(
                    chunk_texts, 
                    batch_size=config.batch_size,
                    show_progress_bar=True
                )
            
            # Convert bfloat16 output to float32 for storage
            chunk_embds = chunk_embds.astype(np.float32)
            
            all_embds[i:end_idx] = chunk_embds
            all_embds.flush()
            
            # Update checkpoint after successful flush
            with open(checkpoint_file, 'w') as f:
                json.dump({'last_completed_idx': end_idx}, f)
            
            elapsed = time.time() - t0
            rate = (end_idx - start_idx) / elapsed
            
            pbar.update(len(chunk_texts))
            pbar.set_postfix({
                'texts/sec': f'{rate:.1f}',
                'chunk': f'{((i-start_idx)//config.save_every + 1)}/{num_chunks}'
            })
            
            wandb.log({
                "texts_processed": end_idx,
                "texts_per_sec": rate,
                "progress": end_idx / len(texts),
            })
        
        pbar.close()
        
        # Stop pool
        if num_gpus > 1:
            model.stop_multi_process_pool(pool)
        
        total_time = time.time() - t0
        
        wandb.log({
            "total_time_minutes": total_time / 60,
            "final_texts_per_sec": (len(texts) - start_idx) / total_time,
        })
        
        print(f"\n✓ Completed {config.model_name}")
        print(f"  Time: {total_time/60:.1f} min | Speed: {(len(texts)-start_idx)/total_time:.1f} texts/sec")
        
        # Remove checkpoint file on successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        del model
        torch.cuda.empty_cache()