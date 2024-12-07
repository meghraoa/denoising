import torch
from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(BaselineModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(8, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return torch.chunk(self.decoder(self.encoder(x)), 2, dim=1)
