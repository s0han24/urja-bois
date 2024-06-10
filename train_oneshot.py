import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import os

transform = transforms.Compose([        # Defining a variable transforms
 transforms.Resize(256),                # Resize the image to 256×256 pixels
 transforms.CenterCrop(224),            # Crop the image to 224×224 pixels about the center
 transforms.ToTensor(),                 # Convert the image to PyTorch Tensor data type
 transforms.Normalize(                  # Normalize the image
 mean=[0.485, 0.456, 0.406],            # Mean and std of image as also used when training the network
 std=[0.229, 0.224, 0.225]      
)])

data_dir = 'test1'

model = models.resnext101_64x4d(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = nn.Sequential(model, nn.Sigmoid())

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

import pandas as pd
import PIL.Image as Image

df = pd.read_csv("image_classification.csv")

def train_one_epoch():
    avg_loss = 0.
    total_loss=0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in df.iterrows():
        # Every data instance is an input + label pair
        input, label = data
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        img = transform(Image.open(input))
        format_img = torch.unsqueeze(img, 0)
        outputs = model(format_img)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, torch.unsqueeze(torch.tensor([float(label)]),0))
        loss.backward()

        print("image {}, loss = {}".format(i, loss.item()))
        total_loss += loss.item()
        # Adjust learning weights
        optimizer.step()
    avg_loss = total_loss/df.shape[0]
    return avg_loss

epoch_number = 0
EPOCHS = 20

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch()
    print('Loss: {}'.format(avg_loss))
    model.train(False)
    epoch_number += 1

# Save the model
torch.save(model, 'model.pt')