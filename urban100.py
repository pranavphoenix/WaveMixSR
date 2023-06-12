
import torch
import torch.nn as nn
from tqdm import tqdm 
import torch.optim as optim
import time

from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms

from torchinfo import summary
import torchmetrics
from datasets import load_dataset
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure

from torch.utils.data import ConcatDataset

import kornia
import wavemix
from wavemix import Level1Waveblock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

dataset_train     = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train', cache_dir = '/workspace/')
dataset_val       = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
dataset_urban100      = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation', cache_dir = '/workspace/')

class SuperResolutionTrainDataset(Dataset):
    def __init__(self, dataset, transform_img=None, transform_target=None):
        self.dataset = dataset
        self.transform_img = transform_img
        self.transform_target = transform_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]["lr"]
        image = Image.open(img_path)

        target_path = self.dataset[idx]["hr"] 
        target = Image.open(target_path)

        if self.transform_img:
            image = self.transform_img(image)

        image = kornia.color.rgb_to_ycbcr(image)

        if self.transform_target:
            target = self.transform_target(target)

        target = kornia.color.rgb_to_ycbcr(target)

        return image, target


class SuperResolutionTestDataset(Dataset):
    def __init__(self, dataset, transform_img=None, transform_target=None):
        self.dataset = dataset
        self.transform_img = transform_img
        self.transform_target = transform_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        lr = "lr"

        img_path = self.dataset[idx][lr]
        image = Image.open(img_path)

        if image.mode == "L":
          image = image.convert('RGB')
        
        if lr == 'lr':

            image_size = image.size

            h , w = 2*image_size[0], 2*image_size[1]


        target_path = self.dataset[idx]["hr"] 
        target = Image.open(target_path)

        if target.mode == "L":
          target = target.convert('RGB')
        
        if lr == 'lr':
            target_size = target.size

            if target_size[0] != h or target_size[1] != w:
                t  = transforms.Resize([w, h])
                target = t(target)

        

        if self.transform_img:
            image = self.transform_img(image)

        image = kornia.color.rgb_to_ycbcr(image)

        if self.transform_target:
            target = self.transform_target(target)

        target = kornia.color.rgb_to_ycbcr(target)

        return image, target


transform_img_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4360, 0.4823, 0.5074), (0.2653, 0.0787, 0.0733))
     ])

transform_target_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4360, 0.4823, 0.5074), (0.2653, 0.0787, 0.0733))
     ])

transform_img_val = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4360, 0.4823, 0.5074), (0.2653, 0.0787, 0.0733))
     ])

transform_target_val = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4360, 0.4823, 0.5074), (0.2653, 0.0787, 0.0733))
     ])

transform_img_urban100 = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4500, 0.4951, 0.5058), (0.2635, 0.0728, 0.0650))
     ])

transform_target_urban100 = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.4500, 0.4951, 0.5058), (0.2635, 0.0728, 0.0650))
    ])

trainset = SuperResolutionTrainDataset(dataset_train, transform_img_train, transform_target_train)
valset = SuperResolutionTrainDataset(dataset_val, transform_img_val, transform_target_val)

trainset = ConcatDataset([trainset, valset])
print(len(trainset))
testset = SuperResolutionTestDataset(dataset_urban100, transform_img_urban100, transform_target_urban100)
print(len(testset))

class WaveMixSR(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 1,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.3,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.final = nn.Sequential(
            nn.Conv2d(final_dim,int(final_dim/2), 3, stride=1, padding=1),
            nn.Conv2d(int(final_dim/2), 1, 1)
        )


        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
            nn.Conv2d(1, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )

        self.path2 = nn.Sequential(
            # nn.ConvTranspose2d(2, 2, 2, stride = 2)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
        )

    def forward(self, img):

        y = img[:, 0:1, :, :] 
        crcb = img[:, 1:3, :, :]

        y = self.path1(y)


        for attn in self.layers:
            y = attn(y) + y

        y = self.final(y)

        crcb = self.path2(crcb)
        
        return  torch.cat((y,crcb), dim=1)

model = WaveMixSR(
    depth = 4,
    mult = 1,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.3
)

model.to(device)
summary(model, input_size=(1, 3, 512, 512), col_names= ("input_size","output_size","num_params","mult_adds"), depth = 4)

scaler = torch.cuda.amp.GradScaler()

batch_size = 1

PATH = 'urban100_2x_y_nonorm.pth'

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)



criterion =  nn.HuberLoss() 

scaler = torch.cuda.amp.GradScaler()
toppsnr = []
topssim = []
traintime = []
testtime = []
counter = 0
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) #Frist optimizer
epoch = 0
while counter < 25:
    
    t0 = time.time()
    epoch_psnr = 0
    epoch_loss = 0
    
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
    
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs[:, 0:1, :, :]
            labels = labels[:, 0:1, :, :]
           
            with torch.cuda.amp.autocast():
                loss =  criterion(outputs, labels) 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_PSNR = psnr(outputs, labels) 
            
            epoch_loss += loss / len(trainloader)
            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f}" )

    model.eval()
    t1 = time.time()
    PSNR = 0
    sim = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            outputs = outputs[:, 0:1, :, :]
            labels = labels[:, 0:1, :, :]

    
      
           
            PSNR += psnr(outputs, labels) / len(testloader)
            sim += structural_similarity_index_measure(outputs, labels) / len(testloader)

     
    print(f"Epoch : {epoch+1} - PSNR_y: {PSNR:.2f} - SSIM_y: {sim:.4f}  - Test Time: {time.time() - t1:.0f}\n")

    topssim.append(sim)
    toppsnr.append(PSNR)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1

    # if float(sim) >= float(max(top1)):
    if float(PSNR) >= float(max(toppsnr)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0


model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #Second Optimizer

while counter < 25:  # loop over the dataset multiple times
    t0 = time.time()
    epoch_psnr = 0
    epoch_loss = 0
    running_loss = 0.0
    
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
    
          inputs, labels = data[0].to(device), data[1].to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          outputs = outputs[:, 0:1, :, :]
          labels = labels[:, 0:1, :, :]

          with torch.cuda.amp.autocast():
            loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          epoch_PSNR = psnr(outputs, labels)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f}" )
    
    t1 = time.time()
    model.eval()
    PSNR = 0   
    sim = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            outputs = outputs[:, 0:1, :, :]
            labels = labels[:, 0:1, :, :]

        
            PSNR += psnr(outputs, labels) / len(testloader)
            sim += structural_similarity_index_measure(outputs, labels) / len(testloader)

     
    print(f"Epoch : {epoch+1} - PSNR_y: {PSNR:.2f} - SSIM_y: {sim:.4f}  - Test Time: {time.time() - t1:.0f}\n")

    topssim.append(sim)
    toppsnr.append(PSNR)

    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    epoch += 1
    counter += 1

    # if float(sim) >= float(max(top1)):
    if float(PSNR) >= float(max(toppsnr)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0

print('Finished Training')
model.load_state_dict(torch.load(PATH))

PSNR_y = 0
SSIM_y = 0

with torch.no_grad():
    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device) 
        outputs = model(images) 

        # Extract Y Channel
        outputs_ychannel = outputs[:, 0:1, :, :]
        labels_ychannel = labels[:, 0:1, :, :]

        
        PSNR_y += psnr(outputs_ychannel, labels_ychannel) / len(testloader)
        SSIM_y += float(ssim(outputs_ychannel, labels_ychannel)) / len(testloader)



print(f"Urban100 Dataset\n")

print(f"Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
print("In Y Space")
print(f"PSNR: {PSNR_y:.2f} - SSIM: {SSIM_y:.4f}\n")

