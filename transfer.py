import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Style Transfer')

parser.add_argument('--pt_path',
                    type=str,
                    default=True,
                    help='location of pt file')

parser.add_argument('--image_path',
                    type=str,
                    default=True,
                    help='location of image')

args = parser.parse_args()

pt_path = args.pt_path
img_path = args.image_path
device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')

# Parameters
batch_size = 64
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
ngpu = 1

# Define for pt
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

#모델 불러오기
checkpoint = torch.load(pt_path)

model = VGG().to(device)
optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

model.load_state_dict(checkpoint['model_state_dict'])
generator.load_state_dict(checkpoint['generator_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

# Load and preprocess the input image
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

input_image = Image.open(img_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Generate a random noise tensor with the desired number of channels
noise = torch.randn(1, nz, 1, 1, device=device)

# Expand the noise tensor to match the shape of the input tensor
noise = noise.expand(input_tensor.size(0), -1, input_tensor.size(2), input_tensor.size(3))

# Generate an image using the generator
with torch.no_grad():
    generated_image = generator(noise)

output_size = input_image.size
output_size = (output_size[1], output_size[0])
generated_image = F.interpolate(generated_image, size=output_size, mode='bilinear', align_corners=False)

# Save the generated image
output_path = './results/output_image.jpg'
save_image(generated_image, output_path)