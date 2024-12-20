import torch
import torch.nn as nn
from ultralytics import YOLO

# Define the custom CNN and attention layers
class CustomCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CustomCNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        squeeze = self.fc1(avg_pool)
        squeeze = self.fc2(squeeze)
        scale = self.sigmoid(squeeze).view(-1, x.size(1), 1, 1)
        return x * scale

# Custom YOLOv8n model with CNN and Attention
class CustomYOLOv8n(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomYOLOv8n, self).__init__()
        self.model = YOLO('yolov8n.pt')  # Load the pretrained YOLOv8n model

        # Custom CNN layers after the main YOLO model (this will modify the backbone output)
        self.custom_cnn1 = CustomCNNLayer(64, 128)  # Adjust channels based on YOLOv8's output
        self.custom_cnn2 = CustomCNNLayer(128, 256)

        # Add attention block after custom CNN layers
        self.attention = SEBlock(256)

    def forward(self, x):
        # Pass through the entire YOLO model (YOLOv8 model is a complete architecture)
        x = self.model(x)  # Forward pass through YOLO model
        
        # Apply custom CNN layers (after the main YOLOv8 model's backbone)
        x = self.custom_cnn1(x)
        x = self.custom_cnn2(x)

        # Apply attention mechanism
        x = self.attention(x)

        # Return final detections
        return x

# Instantiate the model
model = CustomYOLOv8n()

# Hyperparameters
hyperparameters = {
    'epochs': 25,  # Adjust number of epochs
    'batch': 16,  # Adjust batch size based on your GPU
    'img_size': 640,  # Image size (adjust as needed)
    'learning_rate': 0.001,  # Adjust learning rate
    'weight_decay': 0.0005  # L2 regularization
}

# Define dataset path using the data.yaml file
data_yaml_path = 'data.yaml'  # Path to your data.yaml file

# Load the YOLO model
yolo_model = YOLO('yolov8n.pt')

# Train the model using the custom model with custom layers
yolo_model.train(
    data=data_yaml_path,          # Path to the data.yaml
    epochs=hyperparameters['epochs'],
    batch=hyperparameters['batch'],  # Corrected argument name
    imgsz=hyperparameters['img_size'],
    lr0=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay']
)
