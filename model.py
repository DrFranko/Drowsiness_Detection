import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Pooling and Activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Dynamically calculate the flattened size for the fully connected layers
        flattened_size = self._calculate_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)  # Output is 2 for binary classification

    def _calculate_output_size(self):
        # Create a dummy input tensor with the expected image size
        x = torch.zeros(1, 3, 32, 32)  # Assuming input size of 32x32 like in the Keras model
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.adaptive_pool(x)
        flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        # Forward pass through conv layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool(x)
        x = self.dropout3(x)

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)

        return x