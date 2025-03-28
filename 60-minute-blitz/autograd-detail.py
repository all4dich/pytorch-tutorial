import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

#Q.backward(torch.tensor([1., 1.]))
print(torch.tensor([1., 1.]))
print("original Q", Q)
print(9*a**2 == a.grad)
print(-2*b == b.grad)
Q.backward(gradient=torch.tensor([2., 1.]))
print("After backward Q", Q)
print(9*a**2 == a.grad)
print(-2*b == b.grad)
print(a.grad)
print(b.grad)


# weight = weight - learning_rate * gradient
