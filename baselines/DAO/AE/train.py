import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from AE_network import AEModel
from data_process import VideoFrameDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Parameters
learning_rate = 0.0003
weight_decay = 1e-5
num_epochs = 70  # Set the appropriate number of epochs


# Initialize dataset and dataloader
root_dir = '/home/ubuntu/lyra/DAO/dataset/train'
csv_file = os.path.join(root_dir, 'train_scores.csv')
train_dataset = VideoFrameDataset(csv_file=csv_file, root_dir=root_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AEModel(input_features=8400*10, output_features=33)
# model.load_state_dict(torch.load('model.ckpt'))
model.to(device)
model.train()  # Set the model to training mode

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_train = []
# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)
        predictions = model.forward(images)
        loss = criterion(predictions, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # if (i+1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    loss_train.append(epoch_loss/len(train_loader))
    # Save the model checkpoint
    torch.save(model.state_dict(), f'models/ae/model_{epoch}.ckpt')

with open('y_train_loss_epoch.txt', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(loss_train)
    writer.writerow([])


plt.figure()
plt.figure(figsize=(5, 5))
x = range(1, num_epochs + 1)
plt.plot(x, loss_train)
my_x_ticks = np.arange(0, num_epochs + 1, 20)
plt.xticks(my_x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f'y_train_loss.png', dpi=300)

