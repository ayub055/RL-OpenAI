import torch

X = torch.randn(1024, 17, 5)
y = torch.randn(1024, 17, 1)
a = torch.randn(1024, 6)
print(X[0, :, :])
print(X[0, :, -2])
print(a[0,:])
x = X[:, :, -1]
z = torch.cat((x, a), dim=1)
print(z.shape)
print(z[0,:])


# print(y[0, :, :])

# torch.cat(X, a)