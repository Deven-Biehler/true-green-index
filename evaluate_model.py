import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


test_data = "/data/test"
model = "model.ckpt"

# Load the model
model = torch.load(model)
model.eval()

# Load the data
test_data = datasets.ImageFolder(test_data, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Evaluate the model
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# Save the results
with open("results.txt", "w") as f:
    f.write(f"Accuracy: {100 * correct / total}%")

