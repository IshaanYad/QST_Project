import time

import torch

from data import generate_dataset
from model import DensityNet
from utils import fidelity, trace_distance

inputs, targets = generate_dataset(12000)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.complex64)
train_inputs = inputs[:10000]
train_targets = targets[:10000]
test_inputs = inputs[10000:]
test_targets = targets[10000:]
model = DensityNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(train_inputs)
    loss = torch.mean(torch.abs(predictions - train_targets) ** 2)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.6f}")
torch.save(model.state_dict(), "../outputs/model.pt")

fidelities = []
trace_distances = []
start_time = time.time()
with torch.no_grad():
    test_predictions = model(test_inputs)
end_time = time.time()

for i in range(len(test_inputs)):
    fidelities.append(fidelity(test_predictions[i].numpy(), test_targets[i].numpy()))
    trace_distances.append(trace_distance(test_predictions[i], test_targets[i]))
print("Mean Fidelity:", sum(fidelities) / len(fidelities))
print("Mean Trace Distance:", sum(trace_distances) / len(trace_distances))
print("Average Inference Latency:", (end_time - start_time) / len(test_inputs))
mean_fidelity = sum(fidelities) / len(fidelities)
mean_trace_distance = sum(trace_distances) / len(trace_distances)
avg_latency = (end_time - start_time) / len(test_inputs)

with open("../docs/report.md", "w") as f:
    f.write("# Neural Network Based Quantum State Tomography\n\n")

    f.write("## Model Working\n")
    f.write(
        "A neural network was trained to reconstruct a single-qubit density matrix "
        "from simulated Pauli measurement outcomes. Instead of predicting the density "
        "matrix directly, the model outputs a lower-triangular matrix L, and the "
        "density matrix is reconstructed as ρ = LL† / Tr(LL†). "
        "This guarantees physical validity by construction.\n\n"
    )

    f.write("## Training Configuration\n")
    f.write("- System size: 1 qubit\n")
    f.write("- Model: Multilayer Perceptron (MLP)\n")
    f.write("- Training samples: 10000\n")
    f.write("- Test samples: 2000\n")
    f.write("- Epochs: 20\n\n")

    f.write("## Results\n")
    f.write(f"- Mean Fidelity: {mean_fidelity:.4f}\n")
    f.write(f"- Mean Trace Distance: {mean_trace_distance:.4f}\n")
    f.write(f"- Average Inference Latency: {avg_latency:.6f} seconds\n\n")

    f.write("## Replication Guide\n")
    f.write(
        "1. Install dependencies: torch, numpy, scipy\n"
        "2. Run the experiment using: python src/train.py\n"
        "3. Trained model weights are saved in the outputs/ directory\n"
    )
