import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(self, model, device, train_dataset, lr=1e-4, batch_size=1):
        self.model = model.to(device)
        self.device = device
        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for img in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(img)
                loss = self.criterion(outputs, img)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            self.writer.add_scalar('Training Loss', avg_loss, epoch)
        
        self.writer.close()
