import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_data
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join('..', 'data', 'food-101', 'images')
trainloader, classes = load_data(data_dir, batch_size=64)

num_classes = len(classes)
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), '../models/model.pth')
