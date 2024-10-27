import torch
import torchvision.models as models
from torchvision.datasets import FakeData
from torchvision.transforms import transforms
from torchvision.transforms import GaussianBlur, ColorJitter, RandomGrayscale
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.functional import softmax
from torch.optim import Adam
import random
import numpy as np
import torch.nn as nn


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)       
    np.random.seed(seed_value)    
    torch.manual_seed(seed_value)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 


set_seed(42)
num_classes = 20

def modify_model(model, num_classes=20):
    if model.__class__.__name__ == 'GoogLeNet':
        # Adjusting the main classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model.__class__.__name__.startswith('DenseNet'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model.__class__.__name__ == 'EfficientNet':
        # Adjust the last linear layer (classifier) of EfficientNetB0
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)    
    return model

# Example of initializing and modifying models

densenet121 = models.densenet121(weights="DEFAULT")
mobilenet_v2 = models.mobilenet_v2(weights="DEFAULT")
efficientnet_b0 = models.efficientnet_b0(weights="DEFAULT")

#models = [resnet18, resnet50, googlenet, densenet121]
models = [efficientnet_b0]
#model_names = ['ResNet18', 'ResNet50', 'GoogLeNet', 'DenseNet121']
model_names = ['efficientnet_b0']

# Modify each model for 20 classes
modified_models = [modify_model(model) for model in models]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in modified_models:
    model.to(device)


# Synthetic data generation settings

image_size = (3, 224, 224) 

transform_blur_noise = transforms.Compose([transforms.ToTensor(),GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
    transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.3),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_color_variation = transforms.Compose([transforms.ToTensor(),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), RandomGrayscale(p=0.1),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


datasets_blur_noise = [FakeData(size=3000, image_size=image_size, transform=transform_blur_noise, num_classes=num_classes) for _ in range(2)]
datasets_color_variation = [FakeData(size=3000, image_size=image_size, transform=transform_color_variation, num_classes=num_classes) for _ in range(2)]

# Concatenating and splitting datasets for training and testing
train_datasets = []
test_datasets = []
for dataset_list in (datasets_blur_noise, datasets_color_variation):
    for dataset in dataset_list:
        # Splitting each dataset: 1000 for training, 2000 for testing
        train_datasets.append(torch.utils.data.Subset(dataset, indices=range(1000)))
        test_datasets.append(torch.utils.data.Subset(dataset, indices=range(1000, 3000)))

# Concatenate all training subsets and all testing subsets
train_data = ConcatDataset(train_datasets)
test_data = ConcatDataset(test_datasets)

# Creating data loaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False) 



epochs = 20
learning_rate = 0.001


# Function to train and evaluate models
def train_and_evaluate(model, train_loader, test_loader, name):
    optimizer = Adam(model.parameters(), lr=0.001)
    # Train the model
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed for {name}")
    # Evaluate the model
    evaluate_and_save(model, test_loader, name)

# Evaluate and save function
def evaluate_and_save(model, data_loader, file_suffix):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    np.save(f'{file_suffix}_softmax_probs.npy', np.array(all_probs))
    np.save(f'{file_suffix}_true_labels.npy', np.array(all_labels))
    print(f"Saved softmax probabilities and true labels for {file_suffix}.")

# Training and evaluating each model
for model, name in zip(models, model_names):
    train_and_evaluate(model, train_loader, test_loader, name)