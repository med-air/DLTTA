import os
from torch import nn


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True), nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True), nn.Linear(512, 1024))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x