import torch
from torch.utils.data import DataLoader
import os
from AE_network import AEModel
from data_process import VideoFrameDataset

# Parameters
batch_size = 8  # Adjust to your needs

# Initialize dataset and dataloader
root_dir = '../dataset/train'
csv_file = os.path.join(root_dir, 'test_scores.csv')
test_dataset = VideoFrameDataset(csv_file=csv_file, root_dir=root_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AEModel(input_features=8400*85, output_features=33).to(device)
model.load_state_dict(torch.load('model.ckpt'))
model.eval()  # Set the model to evaluation mode

# Test the model
with torch.no_grad():
    total_loss = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model.forward(images)
        loss = torch.nn.functional.mse_loss(predictions, labels)
        total_loss += loss.item()

    print(f'Average MSE Loss on test set: {total_loss / len(test_loader)}')
