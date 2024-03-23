# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 2:43:53 2024

@author: AMAN
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image  

# Define constants
epochs = 5
batch_size = 64
learning_rate = 0.001
best_test_acc = 0  
best_model_path = r"D:\projects\best_model.pth"

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)  
        self.fc3 = nn.Linear(84, 10)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 * 4 * 4) 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x

# Load the datasets
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0  
    total_train = 0
    pbar = tqdm(train_loader, total=len(train_loader))
    for i, data in enumerate(pbar, 0):
        inputs, labels = data
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()  
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        train_acc = 100 * correct_train / total_train
        pbar.set_description('[Epoch %d/%d] Loss: %.3f | Train Accuracy: %.2f%%' %
                             (epoch + 1, epochs, running_loss / (i + 1), train_acc))

    # Validation (test) phase
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()  
    test_acc = 100 * correct_test / total_test
    print('Test Accuracy: %.2f%%' % test_acc)
    
    
    if test_acc > best_test_acc:
        torch.save(model.state_dict(), best_model_path)
        best_test_acc = test_acc

# Load the weights of the best model
best_model = CNN()
best_model.load_state_dict(torch.load(best_model_path))

# Function to make predictions on a single image
def predict_image(image_path, model):
    image = Image.open(image_path).convert('L')  
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),       
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    image_tensor = transform(image).unsqueeze(0) 
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = torch.argmax(output).item()
    return predicted_label

def plot_prediction(image_path, label):
    image = Image.open(image_path).convert('L')  
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Label: {label}")
    plt.axis('off')
    plt.show()
sample_image = r"C:\Users\amand\OneDrive\Documents\8.png"
predicted_label = predict_image(sample_image, best_model)
plot_prediction(sample_image, predicted_label)

