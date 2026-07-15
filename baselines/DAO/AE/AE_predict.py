from PIL import Image
from torchvision import transforms
from .AE_network import AEModel
import torch
import numpy as np
import cv2


model = AEModel(input_features=8400*10, output_features=33)
model.load_state_dict(torch.load('AE/models/model.ckpt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image


def predict(image_path):
    image = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output.cpu().numpy()


# def convert(a, b, p, q):
#     for i in range(q):
#         start_idx = i * p
#         end_idx = start_idx + p
#         b[:, i] = a[start_idx:end_idx]
#

# VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
# resolutions = ['480p', '720p', '1080p']
# button = 33
# re = [[720, 480], [1280, 720], [1920, 1080]]
#
# output = predict('../dataset/Driving/2/00.JPEG')
# output = output[0]
#
# button_list = np.zeros((len(VIDEO_BIT_RATE), len(re)))
# convert(output, button_list, len(VIDEO_BIT_RATE), len(re))
# sub_array = button_list[:4, :]
# max_pos = np.unravel_index(np.argmax(sub_array), sub_array.shape)
# bit = VIDEO_BIT_RATE[max_pos[0]]
# w = re[max_pos[1]][0]
# h = re[max_pos[1]][1]
#
# print(bit, w, h)
