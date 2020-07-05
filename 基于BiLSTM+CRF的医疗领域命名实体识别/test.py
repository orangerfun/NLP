import torch
x = torch.tensor([[1,2],[3,4]], dtype = float)
print("x:\n", x)
print("ssum:\n", torch.logsumexp(x, dim=1))
print("LSE:\n", LSE(x))

def LSE(x):
    x = torch.exp(x)
    x = torch.sum(x, dim = 1)
    return torch.log(x)