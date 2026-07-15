# optical_flow_module.py
"""Optical flow extraction module for video content dynamics"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from config import *


class OpticalFlowExtractor:
    def __init__(self):
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

    def extract_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Extract optical flow between two frames using Farneback method"""
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            self.flow_params['flags']
        )

        return flow

    def extract_flow_from_video_chunk(self, video_chunk: List[np.ndarray], sample_interval: int = 25) -> np.ndarray:
        """Extract optical flow from video chunk at specified intervals"""
        flows = []

        # Sample frames at intervals (e.g., every 25 frames for 1 second at 25fps)
        for i in range(0, len(video_chunk) - sample_interval, sample_interval):
            flow = self.extract_flow(video_chunk[i], video_chunk[i + sample_interval])
            flows.append(flow)

        # Concatenate flows
        if flows:
            return np.concatenate(flows, axis=2)
        else:
            # Return zero flow if not enough frames
            h, w = video_chunk[0].shape[:2]
            return np.zeros((h, w, 2))


class ResNetFlowEncoder(nn.Module):
    """ResNet-based encoder for optical flow features (simplified ResNet-18)"""

    def __init__(self, input_channels=2, output_dim=512):
        super(ResNetFlowEncoder, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling and output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialTemporalFusionBlock(nn.Module):
    """Spatial-Temporal fusion block for combining multimodal features"""

    def __init__(self, flow_dim=512, temporal_dim=128, output_dim=256):
        super(SpatialTemporalFusionBlock, self).__init__()

        # Flow feature processing
        self.flow_fc = nn.Linear(flow_dim, 256)

        # Temporal feature processing (for bandwidth, latency history)
        self.temporal_fc = nn.Linear(temporal_dim, 256)

        # Bilinear pooling for fusion
        self.bilinear = nn.Bilinear(256, 256, output_dim)

        # Final processing
        self.fusion_fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, flow_features, temporal_features):
        # Process individual features
        flow_processed = F.relu(self.flow_fc(flow_features))
        temporal_processed = F.relu(self.temporal_fc(temporal_features))

        # Bilinear fusion
        fused = self.bilinear(flow_processed, temporal_processed)

        # Final processing
        output = self.fusion_fc(fused)

        return output