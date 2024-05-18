import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from utils.visualization import show_batch, show_image
import sys
from einops import rearrange
import matplotlib.pyplot as plt

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc, file=sys.stdout)

class Trainer:
    def __init__(self, model, device, train_dataset, val_dataset, lr=1e-3, batch_size=1):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    def train(self, num_epochs):
        validation_loss = 0
        for epoch in range(num_epochs):
            self.model.train()
            training_loop = create_tqdm_bar(self.train_loader, desc=f'Training Epoch [{epoch}/{num_epochs}]')
            training_loss = 0
            for train_iteration, img_batch in training_loop:
                self.optimizer.zero_grad()
                img_batch = img_batch.to(self.device)
                loss = self.model(img_batch)
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()

                # Update the progress bar.
                training_loop.set_postfix(train_loss = "{:.8f}".format(training_loss / (train_iteration + 1)), val_loss = "{:.8f}".format(validation_loss))
                training_loop.refresh()

                # Update the tensorboard logger.
                self.writer.add_scalar('Training Loss', loss.item(), epoch * len(self.train_loader) + train_iteration)

                #for name, param in self.model.named_parameters():
                 #   if param.grad is not None:
                  #      self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(self.train_loader) + train_iteration)


        self.writer.close()

        with torch.no_grad():
            self.model.eval()
            img = self.model()
            if img.is_cuda:
                img = img.cpu()

            # display greyscale image
            
            plt.imshow(img.squeeze(),cmap='gray')
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.savefig('cat_out.png', bbox_inches='tight', pad_inches=0)
            plt.close()