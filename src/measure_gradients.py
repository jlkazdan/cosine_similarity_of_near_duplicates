import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GradientComputer:
    def __init__(self, model_name, device='cuda', use_flash_attention=False):
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention
        
        # Load model configuration
        model_kwargs = {
            'device_map': "auto",
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True
        }
        
        # Optionally use flash attention 2
        if use_flash_attention:
            model_kwargs['attn_implementation'] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Don't use torch.compile - it causes issues with SDPA kernels
        # If you need compilation, set suppress_errors:
        # import torch._dynamo
        # torch._dynamo.config.suppress_errors = True
        # self.model = torch.compile(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_gradient(self, corpus):
        """Compute gradients for the given text corpus."""
        self.model.train()
        
        # Tokenize with truncation at 2000 tokens
        tokenized_text = self.tokenizer(
            corpus, 
            return_tensors="pt", 
            truncation=True,
            max_length=2000,
            padding=True
        ).to(next(self.model.parameters()).device)
        
        # Forward pass with gradient computation
        try:
            if self.use_flash_attention:
                # Use flash attention if enabled
                outputs = self.model(
                    tokenized_text.input_ids, 
                    labels=tokenized_text.input_ids, 
                    use_cache=False
                )
            else:
                # Standard attention
                outputs = self.model(
                    tokenized_text.input_ids, 
                    labels=tokenized_text.input_ids, 
                    use_cache=False
                )
        except Exception as e:
            print(f"Warning: Error during forward pass: {e}")
            print("Retrying without special attention kernels...")
            # Fallback to eager mode
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                outputs = self.model(
                    tokenized_text.input_ids, 
                    labels=tokenized_text.input_ids, 
                    use_cache=False
                )
        
        # Backward pass
        outputs.loss.backward()

    def return_gradient(self, collect_device='cuda:0'):
        """Return and clear gradients."""
        # Collect all gradients
        all_grad = [
            param.grad.detach().clone() 
            for param in self.model.parameters() 
            if param.grad is not None
        ]
        
        # Clear gradients
        self.model.zero_grad()
        # torch.cuda.empty_cache()
        
        return all_grad