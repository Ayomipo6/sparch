import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=False):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layer_sizes, num_classes, dropout=0.0):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList([CNNLayer(*params) for params in conv_layers])
        self.feature_size = self._get_conv_output(input_shape)

        # Fully connected layers
        fc_layers = []
        in_features = self.feature_size
        for out_features in fc_layer_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

    def _get_conv_output(self, input_shape):
        if None in input_shape:
            raise ValueError("input_shape must be fully specified and contain no None values")
        # Proceed to generate a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Explicitly adding a batch dimension
            output = dummy_input
            for layer in self.conv_layers:
                output = layer(output)
            output_size = output.view(1, -1).size(1)
        return output_size


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)
