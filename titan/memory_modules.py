import torch
import torch.nn as nn
from titan.titan_block import TitanBlock
# from titan.attention_modules import SlidingWindowAttention  # Assume you implement this module.

class MACModule(nn.Module):
    def __init__(self, d_model, persistent_len=10):
        super(MACModule, self).__init__()
        self.titan_block = TitanBlock(d_model)
        self.persistent_mem = nn.Parameter(torch.randn(persistent_len, d_model))
        # Define an attention module for combining persistent mem, retrieved context, and current segment.
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)
    
    def forward(self, current_segment, historical_memory):
        # current_segment: (batch, seg_len, d_model)
        # historical_memory: (batch, hist_len, d_model) retrieved from previous segments
        batch_size = current_segment.size(0)
        persistent = self.persistent_mem.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([persistent, historical_memory, current_segment], dim=1)
        combined = combined.transpose(0, 1)  # for nn.MultiheadAttention
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = attn_output.transpose(0, 1)
        # Optionally update historical memory here
        return attn_output

class MAGModule(nn.Module):
    def __init__(self, d_model):
        super(MAGModule, self).__init__()
        self.sliding_window_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)
        self.memory_module = TitanBlock(d_model)
        self.gate_layer = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Short-term branch using sliding window attention:
        x_sw = x.transpose(0, 1)
        short_term_out, _ = self.sliding_window_attn(x_sw, x_sw, x_sw)
        short_term_out = short_term_out.transpose(0, 1)
        
        # Long-term branch using TitanBlock
        long_term_out, mem_loss = self.memory_module(x)
        
        # Compute gating weights and fuse outputs
        gate = torch.sigmoid(self.gate_layer(x))
        fused = gate * long_term_out + (1 - gate) * short_term_out
        fused = self.output_proj(fused)
        return fused, mem_loss

class MALModule(nn.Module):
    def __init__(self, d_model):
        super(MALModule, self).__init__()
        self.memory_layer = TitanBlock(d_model)
        self.sliding_window_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)
    
    def forward(self, x):
        # Pass input through memory layer
        mem_out, mem_loss = self.memory_layer(x)
        # Apply sliding window attention
        mem_out_sw = mem_out.transpose(0, 1)
        out, _ = self.sliding_window_attn(mem_out_sw, mem_out_sw, mem_out_sw)
        out = out.transpose(0, 1)
        out = out + x  # Residual connection
        return out, mem_loss
