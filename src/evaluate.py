import os
import torch
from data_loader import load_data
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join('..', 'data', 'food-101', 'images')
testloader, classes = load_data(data_dir, batch_size=64, num_samples=100)

num_classes = len(classes)
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load('../models/model.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 100 test images: {100 * correct / total}%')
