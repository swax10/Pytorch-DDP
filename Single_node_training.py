# Pytorch Distributed Training (DDP)

# Numpy for mathematical operations
import numpy as np

# Torch for main operations
import torch

# Torch.nn for defining neural network layers and modules
import torch.nn as nn

# Torch.optim for defining optimization algorithms
import torch.optim as optim

# Torch.utils.data for creating datasets and data loaders
#from torch.utils.data import DataLoader, random_split

# Torch.nn.parallel for implementing distributed training
from torch.nn.parallel import DistributedDataParallel as DDP

# Torch.distributed for managing distributed training
import torch.distributed as dist

# Torch.optim.lr_scheduler for managing learning rate schedules
"""
Learning rate scheduling means adjusting the learning rate based on the number of epochs 
which helps in model convergence and avoiding overfitting

"""
import torch.optim.lr_scheduler as lr_scheduler

# Importing Model from model.py
from model import SimpleNet

# Importing Dataset from dataset.py
from dataset import CustomDataset
import logging
from torch.utils.data import DataLoader, random_split

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Set torch device to GPU, raise an error if GPU is not available as NCCL doesn't support CPU
if not torch.cuda.is_available():
    raise ValueError('NCCL doesn\'t support CPU')

device = torch.device('cuda')

# Initialize the distributed environment
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Custom data augmentation for numerical data
def numerical_augmentation(data):
    noise = np.random.normal(0, 0.1, data.shape)
    return data + noise

# Create model and move it to GPU
model = SimpleNet().to(device)
model = DDP(model, device_ids=[local_rank])

# Create dataset and split into train and validation
dataset = CustomDataset(transform=numerical_augmentation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
# Added weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

best_val_accuracy = 0
patience = 10
no_improve = 0
number_of_epochs = 100

try:
    for epoch in range(number_of_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        # Reduce the accuracy across all processes
        reduced_accuracy = torch.tensor([val_accuracy], device=device)
        dist.all_reduce(reduced_accuracy, op=dist.ReduceOp.SUM)
        reduced_accuracy /= dist.get_world_size()

        if local_rank == 0:
            print(f'Epoch: {epoch+1}, Loss: {epoch_loss/len(train_dataloader):.4f}, '
                  f'Validation Accuracy: {reduced_accuracy.item():.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

            if reduced_accuracy.item() > best_val_accuracy:
                best_val_accuracy = reduced_accuracy.item()
                torch.save(model.state_dict(), f'best_model.pth')
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping")
                break

        # Break the loop if the reduced global accuracy reaches 95%
        if reduced_accuracy.item() >= 0.95:
            break

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
finally:
    if local_rank == 0:
        print('Training completed.')