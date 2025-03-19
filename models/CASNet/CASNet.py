import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils import spectral_norm
from models.CASNet.batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools

from torchvision import transforms
from PIL import Image
from models.CASNet import cadt

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=64, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        w, h = imsize
        for cv in converts:
            self.w[cv] = nn.Parameter(torch.ones(1, ch, h//down_scale, w//down_scale), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles = dict(), dict()
        for key in features.keys():
            styles[key] = self.conv(features[key])  # equals to F - wS(F) see eq.(2)
            contents[key] = features[key] - styles[key]  # equals to wS(F)
            if '2' in key:  # for 3 datasets: source-mid-target
                source, target = key.split('2')
                contents[target] = contents[key]

        if converts is not None:  # separate features of converted images to compute consistency loss.
            for cv in converts:
                source, target = cv.split('2')
                contents[cv] = self.w[cv] * contents[source]
        return contents, styles


class Generator(nn.Module):
    def __init__(self, channels=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )

    def forward(self, content, style):
        return self.model(content+style)


#MNIST Classifier
class Classifier(nn.Module):
    def __init__(self, channels=3, num_classes=2):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(weights='IMAGENET1K_V1').features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out


class Discriminator_Can_test(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_Can_test, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 16, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output


class Discriminator_MNIST(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator_MNIST, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output


class PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(PatchGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class VGG16_classifier(nn.Module):
    def __init__(self):
        super(VGG16_classifier, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(16384, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def freeze_layers(self):
        for name, param in self.named_parameters():
            if 'classifier' not in name and 'features.30' not in name:
                param.requires_grad = False
                
                
class Converter:
    def __init__(self, step, img_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.img_size = img_size
        
        self.nets = {
            'E': Encoder(),
            'G': Generator(),
            'S': Separator((img_size, img_size), ['SC2RC', 'RC2SC']),
        }
        
        for net in self.nets.keys():
            self.nets[net].load_state_dict(torch.load(f'/home/james/MyFolder/code/GIT/CASNet_prj_b/checkpoint/SC2RC/{step}/net{net}.pth'))
            self.nets[net].to(self.device)
            self.nets[net].eval()
    
    def load_image(self, abs_img_path):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        img = Image.open(abs_img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(self.device) # Add batch dimension and move to same device as model
        return img
    
    def convert(self, abs_img_path, CADT=True):
        # Load and preprocess the image
        input_image = self.load_image(abs_img_path).cuda()  # Move to GPU if available

        converts = ['SC2RC', 'RC2SC']
        features = { 'SC': None, 'RC': None }
        
        converted_imgs = {}
        
        # Perform inference (pass through Encoder and Generator)
        with torch.no_grad():  # No need to track gradients during inference
            
            for dset in list(features.keys()):
                features[dset] = self.nets['E'](input_image)
                
            contents, styles = self.nets['S'](features, converts)
                
            for convert in converts:
                source, target = convert.split('2')
                
                if CADT: 
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                    
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
                
        # convert to PIL
        
        for key in converted_imgs.keys():
            converted_imgs[key] = converted_imgs[key].cpu().squeeze(0)
            converted_imgs[key] = (converted_imgs[key] + 1) / 2
            converted_imgs[key] = transforms.ToPILImage()(converted_imgs[key])  
            
        return converted_imgs      
                
        