import torch
import torch.nn as nn


class DensityNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, measurements):
        params = self.layers(measurements)
        l00 = params[:, 0]
        l10_real = params[:, 1]
        l10_imag = params[:, 2]
        l11 = params[:, 3]
        batch_size = measurements.size(0)
        L = torch.zeros(batch_size, 2, 2, dtype=torch.complex64)
        L[:, 0, 0] = l00
        L[:, 1, 0] = l10_real + 1j * l10_imag
        L[:, 1, 1] = l11
        rho = L @ L.conj().transpose(-1, -2)
        trace = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
        rho = rho / trace.view(-1, 1, 1)

        return rho
