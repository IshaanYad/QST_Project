
A neural network was trained to reconstruct a single-qubit density matrix from simulated Pauli measurement outcomes. 
Instead of predicting the density matrix directly, the model outputs a lower-triangular matrix L, and the density matrix is reconstructed as ρ = LL(^dagger) / Tr(LL(^dagger)). 
This guarantees physical validity by construction.

1).Training Configuration
- System size: 1 qubit
- Model: Multilayer Perceptron (MLP)
- Training samples: 10000
- Test samples: 2000
- Epochs: 20

2).Results
- Mean Fidelity: 0.8423
- Mean Trace Distance: 0.3093
- Average Inference Latency: 0.000000 seconds

3).Replication Guide
1. Install dependencies: torch, numpy, scipy
2. Run the experiment using: python src/train.py
3. Trained model weights are saved in the outputs/ directory
