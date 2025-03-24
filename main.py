import torch
import torch.nn as nn
from titan.memory_modules import MACModule, MAGModule, MALModule

def main():
    # Example configuration
    variant = "MAL"  # choose from "MAC", "MAG", or "MAL"
    d_model = 64
    batch_size = 2
    seq_len = 32
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Instantiate chosen module
    if variant == "MAC":
        # For MAC, we need current segment and historical memory.
        # Here we simulate them:
        current_segment = x[:, :16, :]
        historical_memory = x[:, 16:, :]  # dummy historical memory
        model = MACModule(d_model)
        out = model(current_segment, historical_memory)
        print("MAC output shape:", out.shape)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    elif variant == "MAG":
        model = MAGModule(d_model)
        out, mem_loss = model(x)
        print("MAG output shape:", out.shape)
        print("MAG associative memory loss:", mem_loss.item())
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    elif variant == "MAL":
        model = MALModule(d_model)
        out, mem_loss = model(x)
        print("MAL output shape:", out.shape)
        print("MAL associative memory loss:", mem_loss.item())
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        
    else:
        raise ValueError("Unknown variant")
    
if __name__ == "__main__":
    main()
