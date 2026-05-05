import torch
from torch import nn

from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from features.features import FEATURE_COLUMNS

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS

        self.gru = nn.GRU(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]  # ostatni timestep
        out = self.dropout(out)
        out = self.fc(out)

        return out