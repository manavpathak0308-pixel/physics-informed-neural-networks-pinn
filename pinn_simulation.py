import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Neural Network Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t):
        return self.net(t)

# 2. Physics & System Parameters (Newton's Law of Cooling)
T_env = 25.0  # Environmental temperature
T0 = 100.0    # Initial temperature of the object
k = 0.5       # Thermal characteristic constant

# Generate training points (collocation points)
t_train = torch.linspace(0, 10, 100).view(-1, 1).requires_grad_(True)

# 3. Training Loop Setup
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_history = []

# 4. Physics-Informed Training
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Boundary Condition Loss (t = 0, T = T0)
    t_zero = torch.tensor([[0.0]], requires_grad=True)
    T_pred_zero = model(t_zero)
    loss_boundary = torch.mean((T_pred_zero - T0) ** 2)
    
    # Physics Residual Loss (dT/dt + k(T - T_env) = 0)
    T_pred = model(t_train)
    # Using Automatic Differentiation to get dT/dt
    dT_dt = torch.autograd.grad(T_pred, t_train, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    loss_physics = torch.mean((dT_dt + k * (T_pred - T_env)) ** 2)
    
    # Total Composite Loss
    total_loss = loss_boundary + loss_physics
    total_loss.backward()
    optimizer.step()
    
    loss_history.append(total_loss.item())

# 5. Generate and Save Figures
# Chart 1: Loss Convergence
plt.figure()
plt.plot(loss_history, label='Total Loss (Boundary + Physics)')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.title('PINN Optimization Convergence')
plt.legend()
plt.savefig('loss_curve.png', dpi=300)
plt.close()

# Chart 2: Validation vs Analytical Calculus Solution
t_test = np.linspace(0, 10, 100)
T_analytical = T_env + (T0 - T_env) * np.exp(-k * t_test)

with torch.no_grad():
    t_test_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)
    T_pinn = model(t_test_tensor).numpy().flatten()

plt.figure()
plt.plot(t_test, T_analytical, label='Analytical Solution (Calculus)', color='black', linewidth=2)
plt.plot(t_test, T_pinn, '--', label='PINN Prediction (Neural Net)', color='red', linewidth=2)
plt.xlabel('Time (t)')
plt.ylabel('Temperature (T)')
plt.title('Model Prediction vs. Analytical Truth')
plt.legend()
plt.savefig('prediction_graph.png', dpi=300)
plt.close()

print("Simulation complete. Images saved successfully.")
