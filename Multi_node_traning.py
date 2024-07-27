# Pytorch Distributed Training (DDP)

# OS for setting up the environment
import os

# Socket for managing communication among processes
import socket

# Numpy for mathematical operations
import numpy as np

# Torch for main operations
import torch

# Torch.nn for defining neural network layers and modules
import torch.nn as nn

# Torch.optim for defining optimization algorithms
import torch.optim as optim

# Torch.utils.data for creating datasets and data loaders
from torch.utils.data import DataLoader, random_split

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
#import logging
from torch.utils.data import DataLoader, random_split


def setup(rank, world_size):
    """
    Set up the distributed environment for PyTorch distributed training.

    This function initializes the process group for distributed training,
    sets up the master address and port, and broadcasts the IP address
    of the master node to all other nodes.

    Parameters:
    -----------
    rank : int
        The rank of the current process in the distributed training setup.
        Rank 0 is considered the master node.
    
    world_size : int
        The total number of processes participating in the distributed training.

    Returns:
    --------
    None
        This function doesn't return any value, but it sets up the distributed
        environment as a side effect.
    """
    if rank == 0:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        ip_address = torch.tensor(ip_address.encode(), dtype=torch.uint8)
    else:
        ip_address = torch.zeros(15, dtype=torch.uint8)  # Assuming IPv4 address

    dist.broadcast(ip_address, src=0)
    
    master_addr = ip_address.cpu().numpy().tobytes().decode().rstrip('\x00')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment after training is complete.

    This function destroys the process group that was created for distributed training,
    releasing any resources that were allocated for inter-process communication.

    It should be called at the end of the distributed training process to ensure
    proper cleanup and prevent any resource leaks.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    dist.destroy_process_group()

# Custom data augmentation for numerical data
def numerical_augmentation(data):
    noise = np.random.normal(0, 0.1, data.shape)
    return data + noise

def train(rank, world_size):
    def train(rank, world_size):
        """
        Train a neural network model using distributed data parallel (DDP) training.
    
        This function sets up the distributed environment, initializes the model,
        data loaders, loss function, optimizer, and learning rate scheduler. It then
        trains the model for a specified number of epochs or until early stopping
        criteria are met. The function also handles validation, model saving, and
        cleanup of the distributed environment.
    
        Parameters:
        -----------
        rank : int
            The rank of the current process in the distributed training setup.
            Rank 0 is considered the master node.
        
        world_size : int
            The total number of processes participating in the distributed training.
    
        Returns:
        --------
        None
            This function doesn't return any value. It trains the model and saves
            the best model state dict to a file named 'best_model.pth'.
        """
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = SimpleNet().to(device)
    model = DDP(model, device_ids=[rank])

    dataset = CustomDataset(transform=numerical_augmentation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_val_accuracy = 0
    patience = 10
    no_improve = 0
    number_of_epochs = 100

    try:
        for epoch in range(number_of_epochs):
            train_sampler.set_epoch(epoch)
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
            
            reduced_accuracy = torch.tensor([val_accuracy], device=device)
            dist.all_reduce(reduced_accuracy, op=dist.ReduceOp.SUM)
            reduced_accuracy /= world_size

            if rank == 0:
                print(f'Epoch: {epoch+1}, Loss: {epoch_loss/len(train_dataloader):.4f}, '
                      f'Validation Accuracy: {reduced_accuracy.item():.4f}, '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')

                if reduced_accuracy.item() > best_val_accuracy:
                    best_val_accuracy = reduced_accuracy.item()
                    torch.save(model.module.state_dict(), f'best_model.pth')
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print("Early stopping")
                    break

            if reduced_accuracy.item() >= 0.95:
                break

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    finally:
        if rank == 0:
            print('Training completed.')
        cleanup()

if __name__ == "__main__":
    # Get the number of GPUs per node
    gpus_per_node = torch.cuda.device_count()
    
    # Get the rank of the current node and the total number of nodes
    node_rank = int(os.environ['NODE_RANK'])
    num_nodes = int(os.environ['WORLD_SIZE']) // gpus_per_node
    
    # Calculate the global rank and world size
    world_size = gpus_per_node * num_nodes
    rank = node_rank * gpus_per_node + int(os.environ['LOCAL_RANK'])
    
    # Call the train function with the calculated rank and world_size
    train(rank, world_size)