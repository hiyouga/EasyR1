import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=3250):
        super().__init__()
        self.hidden_size = hidden_size

        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # The classifier now takes only the hidden dimension size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, image_embeds):
        # image_embeds shape: [batch_size, num_tokens, hidden_size]
        # or just [num_tokens, hidden_size] if unbatched

        # Add batch dimension if needed
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(0)

        batch_size, num_tokens, hidden_dim = image_embeds.shape

        # Transpose for pooling over token dimension
        x = image_embeds.transpose(1, 2)  # [batch_size, hidden_size, num_tokens]

        # Apply adaptive pooling to get fixed-size representation
        x = self.adaptive_pool(x)  # [batch_size, hidden_size, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_size]

        # Pass through classifier
        logits = self.classifier(x)
        return logits

    def resize_classifier(self, num_classes):
        """Resize the classifier layer to accommodate a different number of classes"""
        old_classifier = self.classifier
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Copy weights for common classes if possible
        if old_classifier.out_features < num_classes:
            self.classifier.weight.data[:old_classifier.out_features] = old_classifier.weight.data
            if old_classifier.bias is not None:
                self.classifier.bias.data[:old_classifier.out_features] = old_classifier.bias.data

        return self.classifier