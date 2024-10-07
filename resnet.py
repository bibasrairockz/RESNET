import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os
import matplotlib.pyplot as plt

##########################
### SET-UP
##########################

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 20
batch_size = 128

# Architecture
num_classes = 10


##########################
### CIFAR-10 DATASET
##########################

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# 1. Direct Identity Residual Block
##########################
### MODEL
##########################

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        #########################
        ### 1st residual block
        #########################
        
        self.block_1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=16,
                                out_channels=3,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(3)
        )
        
        #########################
        ### Fully connected
        #########################        
        self.linear_1 = torch.nn.Linear(3 * 32 * 32, num_classes)

        
    def forward(self, x):
        #########################
        ### 1st residual block
        #########################
        shortcut = x
        x = self.block_1(x)
        x = torch.nn.functional.relu(x + shortcut)
        
        #########################
        ### Fully connected
        #########################
        logits = self.linear_1(x.view(-1, 3 * 32 * 32))
        return logits

    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)
model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# TRAINING
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
        
        ### FORWARD AND BACK PROP
        logits = model(features)
        cost = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 250:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model = model.eval()  # eval mode to prevent updating batchnorm params during inference
    with torch.set_grad_enabled(False):  # save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

# TEST
print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))


##############################
# 2. Resized Residual Block
class ResidualBlock(torch.nn.Module):
    """ Helper Class"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[1],
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1),
                torch.nn.BatchNorm2d(channels[1]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=channels[1],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0),   
                torch.nn.BatchNorm2d(channels[2])
        )

        self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(2, 2),
                                padding=0),
                torch.nn.BatchNorm2d(channels[2])
        )
            
    def forward(self, x):
        shortcut = self.shortcut(x)  # Resizing the shortcut
        block = self.block(x)
        x = torch.nn.functional.relu(block + shortcut)

        return x
    
##########################
### MODEL
##########################

class ResNetCIFAR(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNetCIFAR, self).__init__()
        
        self.residual_block_1 = ResidualBlock(channels=[3, 16, 32])
        self.residual_block_2 = ResidualBlock(channels=[32, 64, 128])
    
        self.linear_1 = torch.nn.Linear(8 * 8 * 128, num_classes)

        
    def forward(self, x):
        out = self.residual_block_1(x)
        out = self.residual_block_2(out)
         
        logits = self.linear_1(out.view(-1, 8 * 8 * 128))
        return logits

    
torch.manual_seed(random_seed)
model_resized = ResNetCIFAR(num_classes=num_classes)
model_resized = model_resized.to(device)

optimizer = torch.optim.Adam(model_resized.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    model_resized = model_resized.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits = model_resized(features)
        cost = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))

    model_resized = model_resized.eval()  # eval mode to prevent updating batchnorm params during inference
    with torch.set_grad_enabled(False):  # save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model_resized, train_loader)))
        
print('Test accuracy: %.2f%%' % (compute_accuracy(model_resized, test_loader)))






# Function to visualize predictions
def visualize_predictions(model, data_loader, num_images=5):
    model.eval()
    images, labels = next(iter(data_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Move data to CPU and convert to NumPy for visualization
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    # Plot images and predictions
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)) * 0.5 + 0.5)  # Unnormalize
        plt.title(f"Pred: {predicted[i]}\nTrue: {labels[i]}")
        plt.axis('off')
    
    plt.show()


# Visualize predictions for Direct Identity Residual Block
print("Visualizing predictions for Direct Identity Residual Block:")
visualize_predictions(model, test_loader)

# Visualize predictions for Resized Residual Block
print("Visualizing predictions for Resized Residual Block:")
visualize_predictions(model_resized, test_loader)

# Create models directory if it doesn't exist
if not os.path.exists('17.0_RESNET/models'):
    os.makedirs('17.0_RESNET/models')

# Save both models
torch.save(model.state_dict(), 'models/direct_identity_residual_block.pth')
torch.save(model_resized.state_dict(), 'models/resized_residual_block.pth')
print("Models saved in 'models' folder.")