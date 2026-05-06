import torch
from torch import nn

from model.gru_model import GRUModel
from utils.dataset import create_dataloader
from utils.sequences import create_sequences

def train_model(model, train_loader, device, lr, epochs=20):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for seqs, labels in train_loader:

            seqs = seqs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(seqs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}")