"""
Coherence Head for Score Prediction
Predicts coherence score for triplet (A, B, C)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoherenceHead(nn.Module):
    """
    Predicts coherence score s(A,B,C) in [0,1]
    
    Input: (z_A, z_B, z_C) concatenated → 384-dim
    Output: scalar score in [0,1]
    """
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        """
        Args:
            embedding_dim: Dimension of each panel embedding (default: 128)
            hidden_dim: Hidden layer dimension
        """
        super(CoherenceHead, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = embedding_dim * 3  # z_A + z_B + z_C
        
        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor, z_C: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z_A: Panel A embedding [batch_size, embedding_dim]
            z_B: Panel B embedding [batch_size, embedding_dim]
            z_C: Panel C embedding [batch_size, embedding_dim]
        
        Returns:
            score: Coherence score [batch_size, 1] in range [0, 1]
        """
        # Concatenate embeddings
        z_concat = torch.cat([z_A, z_B, z_C], dim=1)  # [batch_size, 384]
        
        # Predict score
        score = self.mlp(z_concat)  # [batch_size, 1]
        
        return score
    
    def predict_score(self, z_A: torch.Tensor, z_B: torch.Tensor, z_C: torch.Tensor) -> float:
        """
        Predict single score (for inference)
        
        Args:
            z_A: Panel A embedding [1, embedding_dim] or [embedding_dim]
            z_B: Panel B embedding [1, embedding_dim] or [embedding_dim]
            z_C: Panel C embedding [1, embedding_dim] or [embedding_dim]
        
        Returns:
            score: float value in [0, 1]
        """
        # Ensure batch dimension
        if z_A.dim() == 1:
            z_A = z_A.unsqueeze(0)
        if z_B.dim() == 1:
            z_B = z_B.unsqueeze(0)
        if z_C.dim() == 1:
            z_C = z_C.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            score = self.forward(z_A, z_B, z_C)
        
        return score.item()


if __name__ == '__main__':
    # Test CoherenceHead
    print("Testing CoherenceHead...")
    
    # Create dummy embeddings
    batch_size = 4
    embedding_dim = 128
    
    z_A = torch.randn(batch_size, embedding_dim)
    z_B = torch.randn(batch_size, embedding_dim)
    z_C = torch.randn(batch_size, embedding_dim)
    
    # Create model
    coherence_head = CoherenceHead(embedding_dim=embedding_dim)
    
    # Forward pass
    scores = coherence_head(z_A, z_B, z_C)
    
    print(f"Input shapes: z_A={z_A.shape}, z_B={z_B.shape}, z_C={z_C.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Output scores: {scores.squeeze().tolist()}")
    print(f"All scores in [0,1]: {torch.all((scores >= 0) & (scores <= 1)).item()}")
    
    # Test single prediction
    z_A_single = torch.randn(embedding_dim)
    z_B_single = torch.randn(embedding_dim)
    z_C_single = torch.randn(embedding_dim)
    
    score_single = coherence_head.predict_score(z_A_single, z_B_single, z_C_single)
    print(f"\nSingle prediction: {score_single:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in coherence_head.parameters())
    trainable_params = sum(p.numel() for p in coherence_head.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ CoherenceHead test passed!")

