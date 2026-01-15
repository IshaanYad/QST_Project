import numpy as np
import torch


def fidelity(rho1, rho2):
    sqrt_rho1 = np.linalg.cholesky(rho1)
    M = sqrt_rho1 @ rho2 @ sqrt_rho1.conj().T
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 0)
    return np.sum(np.sqrt(eigvals)) ** 2


def trace_distance(rho1, rho2):
    if isinstance(rho1, torch.Tensor):
        rho1 = rho1.detach().numpy()
    if isinstance(rho2, torch.Tensor):
        rho2 = rho2.detach().numpy()

    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))
