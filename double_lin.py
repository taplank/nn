import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate sample data of form mx+b for now (straight line)
torch.manual_seed(42)
x_1 = torch.linspace(-10, 10, 100).unsqueeze(1)
x_2 = torch.linspace(-15, 5, 100).unsqueeze(1)
x_input = torch.cat((x_1, x_2), dim=1) 
y = 0 * x_1 + 4 * x_2 + 40

# Define the neural network: 2 → 3 → 3 → 1
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = CustomModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(2000):
    y_pred = model(x_input)
    loss = loss_fn(y_pred, y)

    #Do the backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print progress every 10th epoch
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# Plot results
'''plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), y_pred.detach().numpy(), color='red', label='Fitted Line')
plt.legend()
plt.show()'''

with torch.no_grad():
    print("Final weight and bias:")
    print("Weights:", model.layer1.weight.data if hasattr(model, 'layer1') else model.linear.weight.data)
    print("Biases :", model.layer1.bias.data if hasattr(model, 'layer1') else model.linear.bias.data)

