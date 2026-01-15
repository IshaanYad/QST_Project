import numpy as np

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_SET = [PAULI_X, PAULI_Y, PAULI_Z]


def random_density_matrix():
    mat = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = mat @ mat.conj().T
    return rho / np.trace(rho)


def measure_observables(rho):
    return np.array([np.real(np.trace(rho @ op)) for op in PAULI_SET])


def generate_dataset(num_samples=10000):
    measurements = []
    states = []
    for _ in range(num_samples):
        rho = random_density_matrix()
        measurements.append(measure_observables(rho))
        states.append(rho)

    return np.array(measurements), np.array(states)
