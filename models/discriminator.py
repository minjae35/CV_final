"""
GAN Discriminator for Visual Narrative Understanding
Discriminates between real and generated panel embeddings
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator D: z_B -> probability (real/fake)
    Classifies whether a panel embedding is real or generated
    """
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        """
        Args:
            embedding_dim: Dimension of input embeddings (default: 128)
            hidden_dim: Dimension of hidden layer (default: 256)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # MLP: z_B -> hidden -> probability
        # Input: embedding_dim (128)
        # Output: 1 (real/fake probability)
        
        self.network = nn.Sequential(
            # Layer 1: 128 -> 256
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Layer 2: 256 -> 128
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Layer 3: 128 -> 1
            nn.Linear(hidden_dim // 2, 1)
            # No Sigmoid here - use BCEWithLogitsLoss for numerical stability
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
    
    def forward(self, z_b: torch.Tensor) -> torch.Tensor:
        """
        Discriminate whether embedding is real or fake
        
        Args:
            z_b: [batch_size, embedding_dim] - panel embedding
        
        Returns:
            logits: [batch_size, 1] - logits (use with BCEWithLogitsLoss)
        """
        logits = self.network(z_b)  # [batch_size, 1]
        return logits


def test_discriminator():
    """Test Discriminator with dummy data"""
    print("Testing Discriminator...")
    
    # Create discriminator
    discriminator = Discriminator(embedding_dim=128, hidden_dim=256)
    
    # Create dummy data
    batch_size = 4
    z_b_real = torch.randn(batch_size, 128)
    z_b_fake = torch.randn(batch_size, 128)
    
    # Forward pass
    prob_real = discriminator(z_b_real)
    prob_fake = discriminator(z_b_fake)
    
    print(f"Input z_B shape: {z_b_real.shape}")
    print(f"Output probability shape: {prob_real.shape}")
    print(f"Real probabilities: {prob_real.squeeze().detach().numpy()}")
    print(f"Fake probabilities: {prob_fake.squeeze().detach().numpy()}")
    
    # Check output shape
    assert prob_real.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {prob_real.shape}"
    
    # Check output range [0, 1]
    assert torch.all((prob_real >= 0) & (prob_real <= 1)), "Probabilities should be in [0, 1]"
    assert torch.all((prob_fake >= 0) & (prob_fake <= 1)), "Probabilities should be in [0, 1]"
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("✅ Discriminator test passed!")


if __name__ == '__main__':
    test_discriminator()

