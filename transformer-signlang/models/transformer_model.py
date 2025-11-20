"""
Transformer Sign Language Recognition Model
--------------------------------------------
Temporal Transformer Encoder for Sign Language Classification

Model Components:
1. Input Projection Layer
2. Positional Encoding
3. Transformer Encoder Blocks
4. Pooling Layer (GAP/CLS/Last)
5. Classification Head

Author: Transformer Sign Language Project
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding matrisi oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        
        Returns:
            (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSignLanguageClassifier(nn.Module):
    """
    Temporal Transformer Encoder for Sign Language Recognition
    
    Architecture:
        Input (B, T, 258)
          ↓
        Input Projection (B, T, d_model)
          ↓
        Positional Encoding
          ↓
        Transformer Encoder × N
          ↓
        Pooling (GAP/CLS/Last)
          ↓
        Classification Head (B, num_classes)
    
    Args:
        input_dim (int): Keypoint feature dimension (default: 258)
        d_model (int): Transformer embedding dimension (default: 256)
        nhead (int): Number of attention heads (default: 8)
        num_encoder_layers (int): Number of Transformer encoder layers (default: 6)
        dim_feedforward (int): Dimension of feedforward network (default: 1024)
        dropout (float): Dropout rate (default: 0.1)
        num_classes (int): Number of sign classes (default: 3)
        max_seq_length (int): Maximum sequence length (default: 200)
        pooling_type (str): Pooling strategy - 'gap', 'cls', or 'last' (default: 'gap')
    """
    
    def __init__(
        self,
        input_dim=258,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=3,
        max_seq_length=200,
        pooling_type='gap'
    ):
        super(TransformerSignLanguageClassifier, self).__init__()
        
        self.d_model = d_model
        self.pooling_type = pooling_type
        
        # Validate parameters
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        assert pooling_type in ['gap', 'cls', 'last'], f"pooling_type must be 'gap', 'cls', or 'last'"
        
        # [1] Input Projection: (batch, seq, 258) -> (batch, seq, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # [2] Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # [3] Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU aktivasyon (BERT-style)
            batch_first=True,   # Input shape: (batch, seq, feature)
            norm_first=False    # Post-norm (LayerNorm after residual)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)  # Final layer norm
        )
        
        # [4] Pooling Strategy
        if pooling_type == 'cls':
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # [5] Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_length, input_dim=258)
            mask: Padding mask (batch_size, seq_length) 
                  True for padding positions (to be ignored)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        batch_size, seq_length, _ = x.shape
        
        # [1] Input projection
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # [2] Add CLS token if needed
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)
            
            if mask is not None:
                # CLS token için mask genişlet (CLS never masked)
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # [3] Positional encoding
        x = self.pos_encoder(x)
        
        # [4] Transformer encoding
        # Note: src_key_padding_mask - True pozisyonlar IGNORE edilir
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # [5] Pooling
        if self.pooling_type == 'gap':
            # Global Average Pooling
            if mask is not None:
                # Masked positions hariç ortalama al
                mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq, 1)
                x = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
            else:
                x = x.mean(dim=1)  # (batch, d_model)
        
        elif self.pooling_type == 'cls':
            # [CLS] token'ı kullan
            x = x[:, 0, :]  # (batch, d_model)
        
        elif self.pooling_type == 'last':
            # Son pozisyondaki hidden state
            if mask is not None:
                # Her örnek için son valid pozisyonu bul
                lengths = (~mask).sum(dim=1) - 1  # (batch,)
                lengths = lengths.clamp(min=0)  # Ensure non-negative
                x = x[torch.arange(batch_size, device=x.device), lengths, :]
            else:
                x = x[:, -1, :]  # (batch, d_model)
        
        # [6] Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
    
    def get_attention_weights(self, x, mask=None, layer_idx=-1):
        """
        Extract attention weights from a specific layer
        
        Args:
            x: Input tensor (batch_size, seq_length, input_dim)
            mask: Padding mask (batch_size, seq_length)
            layer_idx: Which encoder layer to extract from (-1 for last)
        
        Returns:
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add CLS if needed
        if self.pooling_type == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            if mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Forward through encoder layers and capture attention
        attention_weights = None
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            nonlocal attention_weights
            # output is a tuple: (output, attention_weights) when return_attention=True
            # But TransformerEncoderLayer doesn't return attention by default
            # We need to access self_attn manually
            pass
        
        # For now, return None (full implementation requires modifying forward pass)
        # This is a placeholder for attention visualization
        print("Warning: get_attention_weights requires modified forward pass")
        return None


def print_model_summary(model, input_shape=(32, 200, 258)):
    """
    Print model summary and parameter count
    
    Args:
        model: TransformerSignLanguageClassifier instance
        input_shape: Example input shape (batch, seq_len, features)
    """
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Architecture details
    print(f"\n🏗️  Architecture:")
    print(f"   Input dimension: {model.input_projection.in_features}")
    print(f"   Embedding dimension (d_model): {model.d_model}")
    print(f"   Number of encoder layers: {len(model.transformer_encoder.layers)}")
    print(f"   Feedforward dimension: {model.transformer_encoder.layers[0].linear1.out_features}")
    print(f"   Number of attention heads: {model.transformer_encoder.layers[0].self_attn.num_heads}")
    print(f"   Pooling type: {model.pooling_type}")
    print(f"   Output classes: {model.classifier[-1].out_features}")
    
    # Test forward pass
    print(f"\n🧪 Test Forward Pass:")
    print(f"   Input shape: {input_shape}")
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Create dummy mask (last 20% are padding)
    seq_len = input_shape[1]
    mask = torch.zeros(input_shape[0], seq_len, dtype=torch.bool).to(device)
    mask[:, int(seq_len * 0.8):] = True  # Mark last 20% as padding
    
    with torch.no_grad():
        output = model(dummy_input, mask=mask)
    
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ Forward pass successful!")
    
    print("\n" + "=" * 80)


def create_model(config):
    """
    Create model from config
    
    Args:
        config: TransformerConfig object
    
    Returns:
        model: TransformerSignLanguageClassifier instance
    """
    model = TransformerSignLanguageClassifier(
        input_dim=config.INPUT_DIM,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES,
        max_seq_length=config.MAX_SEQ_LENGTH,
        pooling_type=config.POOLING_TYPE
    )
    
    return model


if __name__ == '__main__':
    """Test the model"""
    print("\n🧪 Testing Transformer Sign Language Classifier...\n")
    
    # Test with different configurations
    configs = [
        {'name': 'Tiny', 'd_model': 128, 'nhead': 4, 'num_layers': 3, 'dim_ff': 512},
        {'name': 'Small', 'd_model': 256, 'nhead': 8, 'num_layers': 4, 'dim_ff': 1024},
        {'name': 'Base', 'd_model': 256, 'nhead': 8, 'num_layers': 6, 'dim_ff': 1024},
    ]
    
    for cfg in configs:
        print(f"\n{'='*80}")
        print(f"Testing {cfg['name']} Configuration")
        print(f"{'='*80}")
        
        model = TransformerSignLanguageClassifier(
            input_dim=258,
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_encoder_layers=cfg['num_layers'],
            dim_feedforward=cfg['dim_ff'],
            dropout=0.1,
            num_classes=3,
            max_seq_length=200,
            pooling_type='gap'
        )
        
        print_model_summary(model, input_shape=(8, 100, 258))
    
    print("\n✅ All model tests completed successfully!")

