#train

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as dset
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.utils as vutils
import numpy as np
import argparse
import datetime
import os

#Loading the model vgg19 that will serve as the base model
model=models.vgg19(pretrained=True).features
#Assigning the GPU to the variable device
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')

parser = argparse.ArgumentParser(description='Style Transfer')

parser.add_argument('--img_path',
                    type=str,
                    default=True,
                    help='path of folder for the images to be trained')

parser.add_argument('--style_path',
                    type=str,
                    default=True,
                    help='path of folder for the images to be feature')

parser.add_argument('--num_epoch',
                    type=int,
                    default=True,
                    help='the number of epoch')

args = parser.parse_args()

# Set the random seed for reproducibility
torch.manual_seed(42)

# Set the paths for the datasets
img_dataroot = args.img_path
style_dataroot = args.style_path

# Set the parameters for training
num_epochs = args.num_epoch
batch_size = 64
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
ngpu = 1

# Define the transforms for the datasets
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the datasets
img_dataset = datasets.ImageFolder(root=img_dataroot, transform=transform)
style_dataset = datasets.ImageFolder(root=style_dataroot, transform=transform)

# Create the dataloaders
img_dataloader = torch.utils.data.DataLoader(
    img_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
style_dataloader = torch.utils.data.DataLoader(
    style_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

# Define the VGG model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        vgg19 = models.vgg19(pretrained=True).features

        # Extract the required layers from the VGG model
        self.model = nn.Sequential()
        for layer_num, layer in enumerate(vgg19):
            self.model.add_module(str(layer_num), layer)
            if str(layer_num) in self.req_features:
                break

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
        return features

def calc_content_loss(gen_feat,orig_feat):
    #calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    
    return torch.mean((gen_feat-orig_feat)**2)


def calc_style_loss(gen, style):
    
    # 128, 64, 64, 64
    
    batch_size, channel, height, width = gen.shape
    
    # 128, 64, w*h
    
    gen_flat = gen.view(batch_size, channel, -1)
    style_flat = style.view(batch_size, channel, -1)
     
    G = torch.bmm(gen_flat, gen_flat.transpose(1, 2))
    A = torch.bmm(style_flat, style_flat.transpose(1, 2))
    
    #print('G', G.shape)
    #print('A', A.shape)
    
    return torch.mean((G - A) ** 2)



# def calc_style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    # batch_size,channel,height,width=gen.shape
    # G = torch.mm(gen.view(batch_size, channel, height * width), gen.view(batch_size, channel, height * width).transpose(1, 2))
    # A = torch.mm(style.view(batch_size, channel, height * width), style.view(batch_size, channel, height * width).transpose(1, 2))
    # G = torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
    # A = torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        

def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss=content_loss=0
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        #extracting the dimensions from the generated image
        content_loss+=calc_content_loss(gen,cont)
        style_loss+=calc_style_loss(gen,style)
    
    #calculating the total loss of e th epoch
    total_loss=alpha*content_loss + beta*style_loss 
    return total_loss


# Initialize the VGG model
model = VGG().to(device).eval()

# Generate random input noise
input_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Initialize the generator model
generator = nn.Sequential(
    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),
    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh()
).to(device)

# Initialize the optimizer
optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(img_dataloader):
        real_images = images.to(device)
        batch_size = real_images.size(0)

        
        # Generate fake images
        fake_images = generator(input_noise[:batch_size])
        
        #print(fake_images.shape)

        # Extract features from the real and style images
        orig_features = model(real_images)
        style_features = model(next(iter(style_dataloader))[0][:batch_size].to(device))
        
        #print(orig_features[0].shape)
        #print(style_features[0].shape)
        

        # Calculate the loss
        alpha=8
        beta=70
        
        vgg_output = model(fake_images)
        
        #print(vgg_output[0].shape)
        
        total_loss = calculate_loss(vgg_output, orig_features, style_features)
        
        

        # Update the generator
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print the loss
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Step [{i+1}/{len(img_dataloader)}], "
                f"Loss: {total_loss.item():.4f}"
            )

#Store pt
folder_path = os.path.join(os.path.dirname(__file__), '.', 'pt')

current_date = datetime.date.today()
pt_name = "model_" + current_date.strftime("%Y%m%d") + ".pt"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

pt_path = os.path.join(folder_path, pt_name)

torch.save({
            'model_state_dict': model.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, pt_path)