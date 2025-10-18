import torch

input = torch.tensor([1, 2, 3, 4, 5, 0, 0])
print("input:", input)

input = input.squeeze()
print("input:", input)

loss_mask = (input != 0)
print("loss_mask:", loss_mask)

X = torch.tensor(input[:-1])
Y = torch.tensor(input[1:])
loss_mask = torch.tensor(loss_mask[1:])
print("X:", X)
print("Y:", Y)
print("loss_mask:", loss_mask)


