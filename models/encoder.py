"""
Multimodal Panel Encoder: ResNet-50 (Image) + DistilBERT (Text)
Combines visual and textual features for narrative understanding
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Dict, List, Optional, Union


class ImageEncoder(nn.Module):
    """
    Image encoder using pretrained ResNet-50
    Extracts 2048-dim feature vector from global pooling
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = 2048):
        super().__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final FC layer, keep up to avg pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [batch_size, 3, 224, 224]
        
        Returns:
            features: [batch_size, 2048]
        """
        features = self.features(x)  # [batch_size, 2048, 1, 1]
        features = features.flatten(1)  # [batch_size, 2048]
        return features


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained DistilBERT
    Extracts 768-dim feature vector from [CLS] token
    """
    def __init__(self, pretrained_model: str = 'distilbert-base-uncased', feature_dim: int = 768, max_length: int = 96):
        super().__init__()
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        
        self.feature_dim = feature_dim
        self.max_length = max_length
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            features: [batch_size, 768]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        features = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        return features
    
    def encode_texts(self, texts: List[str], device: torch.device, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode text strings
        
        Args:
            texts: List of text strings
            device: Device to place tensors
            max_length: Maximum sequence length (uses self.max_length if None)
        
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_length
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device)
        }


class FusionModule(nn.Module):
    """
    Fuses image and text features into a unified panel embedding
    """
    def __init__(self, image_dim: int = 2048, text_dim: int = 768, output_dim: int = 128):
        super().__init__()
        # Concat image + text → 2048 + 768 = 2816
        concat_dim = image_dim + text_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, 2048]
            text_features: [batch_size, 768]
        
        Returns:
            panel_embedding: [batch_size, 128]
        """
        # Concatenate features
        combined = torch.cat([image_features, text_features], dim=1)  # [batch_size, 2816]
        
        # Apply fusion layers
        panel_embedding = self.fusion(combined)  # [batch_size, 128]
        
        return panel_embedding


class MultimodalEncoder(nn.Module):
    """
    Complete multimodal encoder for comic panels
    Combines ResNet-50 (image) + DistilBERT (text) → unified embedding
    """
    def __init__(
        self,
        image_pretrained: bool = True,
        text_model: str = 'distilbert-base-uncased',
        embedding_dim: int = 128,
        text_max_length: int = 96,
        freeze_image_layers: int = 2,
        freeze_text_layers: int = 2
    ):
        super().__init__()
        
        self.image_encoder = ImageEncoder(pretrained=image_pretrained, feature_dim=2048)
        self.text_encoder = TextEncoder(pretrained_model=text_model, feature_dim=768, max_length=text_max_length)
        self.fusion = FusionModule(image_dim=2048, text_dim=768, output_dim=embedding_dim)
        
        self.embedding_dim = embedding_dim
        
        # Freeze pretrained layers
        if freeze_image_layers > 0:
            self._freeze_image_layers(freeze_image_layers)
        if freeze_text_layers > 0:
            self._freeze_text_layers(freeze_text_layers)
    
    def _freeze_image_layers(self, num_layers: int):
        """Freeze first N ResNet blocks (layer1, layer2, etc.)"""
        # ResNet-50 structure in features: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # We want to freeze layer1, layer2, etc.
        # Get the original ResNet model to access layers directly
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        resnet_children = list(resnet.children())
        
        # Find layer blocks (layer1, layer2, layer3, layer4 are at indices 4, 5, 6, 7)
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i in range(min(num_layers, len(layer_names))):
            layer_name = layer_names[i]
            # Find corresponding layer in our features module
            # Our features module has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
            layer_idx = i + 4  # Skip conv1, bn1, relu, maxpool
            if layer_idx < len(list(self.image_encoder.features.children())):
                layer_module = list(self.image_encoder.features.children())[layer_idx]
                for param in layer_module.parameters():
                    param.requires_grad = False
    
    def _freeze_text_layers(self, num_layers: int):
        """Freeze first N DistilBERT transformer layers"""
        # DistilBERT has 6 transformer layers
        for i in range(min(num_layers, 6)):
            for param in self.text_encoder.bert.transformer.layer[i].parameters():
                param.requires_grad = False
        
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode panels into unified embeddings
        
        Args:
            images: [batch_size, 3, 224, 224]
            texts: List of OCR text strings (if not pre-tokenized)
            input_ids: [batch_size, seq_len] (if pre-tokenized)
            attention_mask: [batch_size, seq_len] (if pre-tokenized)
        
        Returns:
            embeddings: [batch_size, 128]
        """
        # Encode image
        image_features = self.image_encoder(images)  # [batch_size, 2048]
        
        # Encode text
        if texts is not None:
            # Tokenize texts
            encoded = self.text_encoder.encode_texts(texts, device=images.device)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        text_features = self.text_encoder(input_ids, attention_mask)  # [batch_size, 768]
        
        # Fuse features
        embeddings = self.fusion(image_features, text_features)  # [batch_size, 128]
        
        return embeddings
    
    def encode_panels(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Convenience method for encoding panels
        
        Args:
            images: [batch_size, 3, 224, 224]
            texts: List of OCR text strings
        
        Returns:
            embeddings: [batch_size, 128]
        """
        return self.forward(images, texts=texts)


if __name__ == "__main__":
    # Test code
    print("Testing MultimodalEncoder...")
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["HELLO WORLD!", "This is a test.", "[EMPTY]", "Some OCR text here."]
    
    # Initialize encoder
    encoder = MultimodalEncoder(
        image_pretrained=False,  # Use random weights for testing
        embedding_dim=128
    )
    
    # Forward pass
    embeddings = encoder.encode_panels(images, texts)
    
    print(f"Input images shape: {images.shape}")
    print(f"Input texts: {texts}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embedding dim: {encoder.embedding_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

