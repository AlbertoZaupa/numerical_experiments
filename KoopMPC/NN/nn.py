import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the network.
class HankelNetConv(nn.Module):
    def __init__(self, n, n_prime, m, N, hidden_dim):
        """
        n: dimension of x0 (and also the output dimension of phi and each block of the output)
        m: dimension of each control input u
        N: number of control steps (so U is of dimension N*m)
        hidden_dim: hidden dimension for the phi subnetwork.
        """
        super(HankelNetConv, self).__init__()
        self.n = n
        self.m = m
        self.N = N

        # Define the phi subnetwork (only depends on x0)
        hidden_dim = [n] + hidden_dim
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dim)-1):
            self.layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim[-1], n_prime))

        self.phi = nn.Sequential(*self.layers)

        # L: maps phi(x0) (n) to an output of size N*n
        self.L = nn.Linear(n_prime, N * n, bias=False)

        # Instead of a ParameterList for g_params, we use a 1D convolution.
        # This convolution layer will act over the sequence dimension of U.
        # We set:
        #   in_channels = m   (each control input dimension)
        #   out_channels = n  (each output block dimension)
        #   kernel_size = N   (length of the control sequence)
        #
        # Note: to enforce causality (only past inputs affect the current output),
        # we pad the input on the left by (kernel_size - 1).
        self.conv = nn.Conv1d(in_channels=m, out_channels=n, kernel_size=N, bias=False)

        # Optionally, you might want to initialize self.conv.weight in a way that
        # mimics the Hankel structure (e.g., by ensuring that the weight at future
        # time steps is zero). One way is to apply a mask to the convolutional kernel
        # during the forward pass.

    def forward(self, x0, U):
        """
        x0: tensor of shape (batch, n)
        U: tensor of shape (batch, N*m)
        Returns:
            output: tensor of shape (batch, N*n)
        """
        batch_size = x0.shape[0]

        # Process x0 through the phi subnetwork and L
        phi_out = self.phi(x0)  # shape: (batch, n)
        L_term = self.L(phi_out)  # shape: (batch, N*n)

        # Reshape U: from (batch, N*m) to (batch, m, N)
        U_seq = U.view(batch_size, self.N, self.m).permute(0, 2, 1)

        # For causal convolution, pad on the left with (kernel_size - 1) zeros.
        padding = self.N - 1
        U_padded = F.pad(U_seq, (padding, 0))

        # Compute the convolution. The output will have shape (batch, n, L) with L = N.
        conv_out = self.conv(U_padded)

        # Reshape conv_out to (batch, N*n) to match the shape of L_term.
        conv_out = conv_out.permute(0, 2, 1).contiguous().view(batch_size, self.N * self.n)

        return L_term + conv_out

    def get_L(self):
        """ Return the weight matrix L of shape (N*n, n_prime)"""
        return self.L.weight.detach().cpu().numpy()

    def get_G(self):
        """
        Return the full lower-triangular block Hankel matrix G of shape (N*n, N*m).
        Each block is of size (n, m), and there are N blocks per row and column.
        """
        # Extract convolution kernel: shape (n, m, N)
        kernel = self.conv.weight.detach()  # shape: (n, m, N)

        # Reorder dimensions to match the Hankel construction
        # kernel[:, :, i] corresponds to g_{i+1}, shape: (n, m)
        g_blocks = [kernel[:, :, i] for i in range(self.N)]  # g_1, g_2, ..., g_N
        print(g_blocks[0].shape)

        # Construct the block Hankel matrix G
        # Each row has increasing number of blocks: [g_i, g_{i-1}, ..., g_1]
        G = torch.zeros((self.N * self.n, self.N * self.m))
        for i in range(self.N):
            row_start = i * self.n
            col_start = 0
            for j in range(self.N-i):
                G[row_start:row_start + self.n, col_start:col_start + self.m] = g_blocks[i]
                row_start = row_start + self.n
                col_start = col_start + self.m

        return G.cpu().numpy()


# A training loop utility.
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x0, U, target in dataloader:
            x0, U, target = x0.to(device), U.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(x0, U)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x0.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model
