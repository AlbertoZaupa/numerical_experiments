from nn import HankelNetConv
import torch
import matplotlib.pyplot as plt
from pykoopman.common.examples import vdp_osc, rk4

def validate_vdp(nn, dT, n_int, N):
    n_states = 2
    n_inputs = 1
    # Uniform random distributed forcing in [-1, 1]
    u = 2 * torch.rand((n_int, 1), dtype = torch.float) - 1

    # Uniform distribution of initial conditions
    # 2 is the number of state components
    x = 2 * torch.rand((n_states, 1), dtype = torch.float) - 1

    X_true = torch.zeros((n_states * (n_int-N), 1))
    X_pred = torch.zeros(((n_int-N), N * n_states))

    # Integrate
    for step in range(n_int-N):
        X_pred[step:step + 1, :] = nn(x.T.float(), u[step:step + N, :].T)
        y = rk4(0, x, u[step,], dT, vdp_osc)
        X_true[n_states * step: n_states * (step + 1), :] = x
        x = y

    plot_signal_comparisons(X_true, X_pred)


def plot_signal_comparisons(y_tensor: torch.Tensor, predictions_tensor: torch.Tensor, max_offset: int = 5):
    """
    Plots comparisons between the actual values of the first component of y(t)
    and the predicted values of the first component for offsets from 1 to max_offset.

    Args:
        y_tensor (torch.Tensor): A tensor of shape (2*n, 1) containing actual values of the signal.
        predictions_tensor (torch.Tensor): A tensor of shape (n, 2*N) containing predicted values.
        max_offset (int): The maximum offset to plot (default is 5).
    """
    # Determine the number of samples (n)
    n = y_tensor.shape[0] // 2

    # Extract the actual values of the first component y1(t)
    actual_y1 = y_tensor[1::2].view(-1).cpu().detach().numpy()  # shape: (n,)

    # Loop over prediction offsets from 1 to max_offset
    for k in range(1, max_offset + 1):
        # Extract predicted y1(t+k)
        pred_y1 = predictions_tensor[:, 2 * (k - 1) + 1].view(-1).cpu().detach().numpy()
        time_steps = torch.arange(n - k)  # Time steps for predictions

        # Create a new figure
        plt.figure()

        # Scatter plot of actual vs predicted values
        plt.plot(time_steps, actual_y1[k:], label='Actual y1(t)', marker='o', linestyle='-', alpha=0.7)
        plt.plot(time_steps, pred_y1[:-k], label=f'Predicted y1(t+{k})', marker='x', linestyle='--',
                 alpha=0.7)

        # Labels and title
        plt.xlabel("Time step")
        plt.ylabel("y1 value")
        plt.title(f"Actual y1(t) and Predicted y1(t+{k}) over time")
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.show()


if __name__ == '__main__':
    phi = HankelNetConv(n=2, n_prime=7, m=1, N=20, hidden_dim=[64, 64, 32])
    phi.load_state_dict(torch.load('./model.pth'))
    phi.eval()
    validate_vdp(phi, 0.01, 100, 20)