from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attn_weights

class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class PatchEmbedding(nn.Module):
    
    def __init__(self, seq_len: int, patch_size: int, d_model: int, n_vars: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        
        self.projection = nn.Linear(patch_size * n_vars, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = rearrange(x, 'b (n p) v -> b n (p v)', p=self.patch_size)
        x = self.projection(x)  # [batch_size, n_patches, d_model]
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x += self.pos_embedding
        
        return x

class AdaptiveNormalization(nn.Module):
    
    def __init__(self, n_vars: int):
        super().__init__()
        self.n_vars = n_vars
        
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.means = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - self.means) / self.stdev
            return x
        elif mode == 'denorm':
            x = x * self.stdev + self.means
            return x

class ModernTimeSeriesModel(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.n_vars = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.n_layers = getattr(configs, 'n_layers', 6)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.patch_size = getattr(configs, 'patch_size', 16)
        
        if hasattr(configs, 'use_patch_embedding') and configs.use_patch_embedding:
            self.input_embedding = PatchEmbedding(self.seq_len, self.patch_size, 
                                                self.d_model, self.n_vars)
            self.use_patch = True
        else:
            self.input_projection = nn.Linear(self.n_vars, self.d_model)
            self.pos_encoding = PositionalEncoding(self.d_model, self.seq_len + self.pred_len)
            self.use_patch = False
        
        self.adaptive_norm = AdaptiveNormalization(self.n_vars)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self._init_task_specific_layers(configs)
        
        self._init_weights()
        
    def _init_task_specific_layers(self, configs):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.forecast_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.pred_len * self.n_vars)
            )
            
        elif self.task_name == 'imputation':
            self.imputation_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.n_vars)
            )
            
        elif self.task_name == 'anomaly_detection':
            self.anomaly_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.n_vars)
            )
            
        elif self.task_name == 'classification':
            self.classification_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, configs.num_class)
            )
            
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def _encode(self, x):
        if self.use_patch:
            x = self.input_embedding(x)
        else:
            x = self.input_projection(x)
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        attn_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attn_weights_list.append(attn_weights)
            
        return x, attn_weights_list
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, M = x_enc.shape
        
        x_enc = self.adaptive_norm(x_enc, mode='norm')
        
        enc_out, _ = self._encode(x_enc)
        
        if self.use_patch:
            pred_out = self.forecast_head(enc_out[:, 0, :])  # [B, pred_len * n_vars]
            pred_out = pred_out.view(B, self.pred_len, M)
        else:
            pred_out = self.forecast_head(enc_out[:, -1, :])  # [B, pred_len * n_vars]
            pred_out = pred_out.view(B, self.pred_len, M)
        
        pred_out = self.adaptive_norm(pred_out, mode='denorm')
        
        return pred_out
    
    def imputation(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, M = x_enc.shape
        
        if mask is not None:
            means = torch.sum(x_enc * mask, dim=1) / torch.sum(mask, dim=1)
            means = means.unsqueeze(1)
            x_enc_centered = (x_enc - means) * mask
            stdev = torch.sqrt(torch.sum(x_enc_centered ** 2, dim=1) / 
                             torch.sum(mask, dim=1) + 1e-5).unsqueeze(1)
            x_enc = x_enc_centered / stdev
        else:
            x_enc = self.adaptive_norm(x_enc, mode='norm')
        
        enc_out, _ = self._encode(x_enc)
        
        imp_out = self.imputation_head(enc_out)  # [B, L, M]
        
        if mask is not None:
            imp_out = imp_out * stdev + means
        else:
            imp_out = self.adaptive_norm(imp_out, mode='denorm')
        
        return imp_out