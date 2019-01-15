import torch
torch.manual_seed(1991)
from torch.optim import RMSprop, SGD, Adam
from santa import Santa
import matplotlib.pyplot as plt

device = torch.device('cpu')
theta = torch.randn(1, 1, device=device, requires_grad=True)
theta.data = torch.Tensor([[4.0]])

N = 5000
optimizer = Santa(
    [theta],
    lr=1e-1, alpha=0.99, eps=1e-8, weight_decay=0,
    momentum=0.0, centered=False, decay_grad=0.0,
    anne_rate=0.5, burnin=N / 2, N=N)
# optimizer = Adam([theta], lr=1e-2, betas=[0.99, 0.9])
thetas = []

for t in range(N):

    loss = (theta + 4) * (theta + 1) * (theta - 1) * (theta - 3) / 14 + 0.5
    thetas.append(theta.item())
    print(t, theta.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(thetas)
# plt.ylim([-6, 4])
plt.show()
