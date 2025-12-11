"""
Infilling Generator for Visual Narrative Understanding
Generates middle panel embedding from A and C panel embeddings
Improved with Attention mechanism and Residual connections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross-attention module for z_A and z_C interaction"""
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: query attends to key-value
        
        Args:
            query: [batch_size, embedding_dim]
            key: [batch_size, embedding_dim]
            value: [batch_size, embedding_dim]
        
        Returns:
            attended: [batch_size, embedding_dim]
        """
        residual = query
        batch_size = query.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
        K = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)    # [B, H, D]
        V = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)  # [B, H, H]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, V)  # [B, H, D]
        attended = attended.view(batch_size, self.embedding_dim)  # [B, D]
        
        # Output projection and residual
        output = self.out_proj(attended)
        output = self.layer_norm(output + residual)
        
        return output


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.layer_norm(out + residual)
        return out


class InfillingGenerator(nn.Module):
    """
    Improved Generator G: z_A, z_C -> z_B_hat
    Uses cross-attention and residual connections for better performance
    """
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, num_heads: int = 4):
        """
        Args:
            embedding_dim: Dimension of input/output embeddings (default: 128)
            hidden_dim: Dimension of hidden layer (default: 256)
            num_heads: Number of attention heads (default: 4)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.proj_a = nn.Linear(embedding_dim, embedding_dim)
        self.proj_c = nn.Linear(embedding_dim, embedding_dim)
        
        # Cross-attention modules (bidirectional)
        # z_A attends to z_C, z_C attends to z_A
        self.attn_a_to_c = CrossAttention(embedding_dim, num_heads)
        self.attn_c_to_a = CrossAttention(embedding_dim, num_heads)
        
        # Combine attended features
        self.combine = nn.Linear(2 * embedding_dim, hidden_dim)
        
        # Deep MLP with residual connections
        self.mlp_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=0.1),
            ResidualMLPBlock(hidden_dim, hidden_dim * 2, dropout=0.1),
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout=0.1),
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z_a: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        """
        Generate middle panel embedding from A and C
        
        Args:
            z_a: [batch_size, embedding_dim] - embedding of panel A
            z_c: [batch_size, embedding_dim] - embedding of panel C
        
        Returns:
            z_b_hat: [batch_size, embedding_dim] - generated embedding of panel B
        """
        # Project inputs
        z_a_proj = self.proj_a(z_a)
        z_c_proj = self.proj_c(z_c)
        
        # Cross-attention: z_A attends to z_C, z_C attends to z_A
        z_a_attended = self.attn_a_to_c(z_a_proj, z_c_proj, z_c_proj)  # z_A queries z_C
        z_c_attended = self.attn_c_to_a(z_c_proj, z_a_proj, z_a_proj)  # z_C queries z_A
        
        # Combine attended features
        z_combined = torch.cat([z_a_attended, z_c_attended], dim=1)  # [B, 2*embedding_dim]
        z_combined = self.combine(z_combined)  # [B, hidden_dim]
        
        # Pass through residual MLP blocks
        z_features = z_combined
        for mlp_block in self.mlp_blocks:
            z_features = mlp_block(z_features)
        
        # Output projection
        z_b_hat = self.output_proj(z_features)  # [B, embedding_dim]
        
        return z_b_hat


def test_generator():
    """Test InfillingGenerator with dummy data"""
    print("Testing Improved InfillingGenerator...")
    
    # Create generator
    generator = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4)
    
    # Create dummy data
    batch_size = 4
    z_a = torch.randn(batch_size, 128)
    z_c = torch.randn(batch_size, 128)
    
    # Forward pass
    z_b_hat = generator(z_a, z_c)
    
    print(f"Input z_A shape: {z_a.shape}")
    print(f"Input z_C shape: {z_c.shape}")
    print(f"Output z_B_hat shape: {z_b_hat.shape}")
    
    # Check output shape
    assert z_b_hat.shape == (batch_size, 128), f"Expected shape ({batch_size}, 128), got {z_b_hat.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("✅ Improved InfillingGenerator test passed!")


if __name__ == '__main__':
    test_generator()

