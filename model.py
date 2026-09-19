import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """
    Baseline Fully Connected Neural Network (MLP).
    Flattens 1x28x28 input into a 784 vector.
    """
    def __init__(self):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Returns raw logits
        return x


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for MNIST digit classification.
    Architecture:
    [Conv2D -> BatchNorm -> ReLU -> MaxPool] x 2 -> Dropout -> FC -> FC
    """
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Conv Block 1: Input (1, 28, 28) -> Output (32, 28, 28) -> MaxPool -> (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: Input (32, 14, 14) -> Output (64, 14, 14) -> MaxPool -> (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(0.25)
        
        # Fully Connected Layers: Flattened size = 64 channels * 7 * 7 spatial dimensions
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Fully Connected Classification Head
        x = x.view(x.size(0), -1)  # Flatten: [Batch, 64*7*7]
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Returns raw logits
        return x

if __name__ == "__main__":
    # Sanity check with dummy input tensor
    dummy_input = torch.randn(32, 1, 28, 28)
    cnn = DigitCNN()
    output = cnn(dummy_input)
    print(f"CNN Output shape: {output.shape}")  # Expecting [32, 10]