import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# dataset input / output : Just have tensor objects
# w, b
#  : model parameters that have requires_grad=True to track computation
#  : to be optimized
# x : input data
# z : model output
# loss : loss value computed using the model output
# loss.backward() : compute the gradient of loss w.r.t all tensors with requires_grad=True
# Gradient : Of loss function
loss.backward()
print(f"Gradient of w: {w.grad}")
print(f"Gradient of b: {b.grad}")