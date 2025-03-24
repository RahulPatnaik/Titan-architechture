import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                                   padding=padding, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class TitanBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(TitanBlock, self).__init__()
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.dwconv_q = DepthwiseSeparableConv1D(d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.dwconv_k = DepthwiseSeparableConv1D(d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.dwconv_v = DepthwiseSeparableConv1D(d_model, kernel_size=kernel_size, padding=kernel_size//2)
        
        self.gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.memory_transform = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        
        Q = self.act(self.linear_q(x))
        K = self.act(self.linear_k(x))
        V = self.act(self.linear_v(x))
        
        # Prepare for convolution (transpose to (batch, channels, seq_len))
        Q, K, V = [t.transpose(1, 2) for t in (Q, K, V)]
        Q = self.dwconv_q(Q).transpose(1, 2)
        K = self.dwconv_k(K).transpose(1, 2)
        V = self.dwconv_v(V).transpose(1, 2)
        
        # Normalize Q and K using L2 norm
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        d_k = Q.size(-1)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        gate = torch.sigmoid(self.gate(attn_output))
        gated_output = gate * attn_output
        out = self.out_proj(gated_output) + residual
        out = self.norm(out)
        
        mem_pred = torch.matmul(K, self.memory_transform)
        mem_loss = F.mse_loss(mem_pred, V)
        
        return out, mem_loss
