import torch
import numpy as np

data = [[1,2], [3,4]]
print(data)
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)

x_rand = torch.rand_like(x_data, dtype=torch.float)


print(x_data)
print(x_np)
print(x_np.mul(x_data))
print(torch.accelerator.is_available())
print(torch.accelerator.current_accelerator())

tensor = torch.ones(4,4)
print(tensor)

t1 = torch.cat([tensor, tensor],dim=-2)
print(t1)
print(t1.shape)
t1.add_(torch.rand(1))
print(t1 )

