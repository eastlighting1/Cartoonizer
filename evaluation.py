#Perceptual Loss

import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='Style Transfer')

parser.add_argument('--stylized_path',
                    type=str,
                    default=True,
                    help='path of transfered image')

parser.add_argument('--style_path',
                    type=str,
                    default=True,
                    help='path of folder for the images to be feature')

args = parser.parse_args()

def preprocess_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Resize image to match the input size of the style transfer model
    target_size = (224, 224)  # Adjust this size based on your model's input requirements
    image = cv2.resize(image, target_size)

    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def calculate_average_perceptual_loss(stylized_image, target_style_folder):
    # Load pre-trained VGG model
    vgg = models.vgg19(pretrained=True).features
    vgg = nn.Sequential(*list(vgg.children())[:35]).eval()  # Truncate to desired feature layer

    # Convert stylized image to tensor
    stylized_tensor = torch.from_numpy(stylized_image).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Normalize stylized image using VGG normalization
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    stylized_tensor = (stylized_tensor - vgg_mean) / vgg_std

    total_loss = 0
    count = 0

    for file in os.listdir(target_style_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            # Load and preprocess each target style image
            target_style_image_path = os.path.join(target_style_folder, file)
            target_style_image = preprocess_image(target_style_image_path)

            # Convert target style image to tensor
            target_style_tensor = torch.from_numpy(target_style_image).unsqueeze(0).permute(0, 3, 1, 2).float()

            # Normalize target style image using VGG normalization
            target_style_tensor = (target_style_tensor - vgg_mean) / vgg_std

            # Pass stylized and target style images through VGG network
            stylized_features = vgg(stylized_tensor)
            target_style_features = vgg(target_style_tensor)

            # Calculate perceptual loss between feature maps
            criterion = nn.MSELoss()
            perceptual_loss = criterion(stylized_features, target_style_features)

            total_loss += perceptual_loss.item()
            count += 1

    average_loss = total_loss / count
    return average_loss

stylized_image_path = args.stylized_path
target_style_folder = args.style_path

stylized_image = preprocess_image(stylized_image_path)
average_loss = calculate_average_perceptual_loss(stylized_image, target_style_folder)
print(f"Average Perceptual Loss: {average_loss}")
