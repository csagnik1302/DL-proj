import torch

weights=torch.load(r"C:\project\DL project\model\saved_weights\1.pth",map_location="cpu")

print(weights)