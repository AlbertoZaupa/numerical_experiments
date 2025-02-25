import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import VdPDataset
from nn import HankelNetConv, train_model
import torch
import torch.nn as nn

if __name__ == '__main__':
    # Dimensions
    n = 2  # Dimension of x0 and of phi(x0)
    n_prime = 7
    m = 1  # Dimension of each control input u
    N = 20  # Number of control inputs

    # Create dataset and DataLoader.
    dataset = VdPDataset(N=20, n_int=1000, n_traj=200, dT=0.01)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create the model, loss function, and optimizer.
    model = HankelNetConv(n=n, n_prime=n_prime, m=m, N=N, hidden_dim=[64, 64, 32])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model.
    model = train_model(model, dataloader, criterion, optimizer, num_epochs=200)
    torch.save(model.state_dict(), './model.pth')
