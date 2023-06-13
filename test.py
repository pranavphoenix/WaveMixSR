import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms
from torchinfo import summary
import torchmetrics
from datasets import load_dataset
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import wavemix
import kornia

from wavemix import Level1Waveblock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))



class WaveMix(nn.Module):
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False),
            # nn.ConvTranspose2d(2, 2, 2, stride = 2)
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

model = WaveMix(
    depth = 4,
    mult = 1,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.3
)

model.to(device)
summary(model, input_size=(1, 3, 256,256), col_names= ("input_size","output_size","num_params","mult_adds"), depth = 4)



class SuperResolutionDataset(Dataset):
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




# transform_img = transforms.Compose(
#         [#transforms.Resize([256,256], interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.ToTensor(),    
#      ])

# transform_target = transforms.Compose(
#         [#transforms.Resize([512,512], interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.ToTensor(),   
#      ])

resolution = 'original'

if resolution == '256/512':
    transform_img_set5 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5076, 0.4057, 0.2967), (0.3144, 0.2711, 0.2532))
     ])

    transform_target_set5 = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize((0.5076, 0.4057, 0.2967), (0.3144, 0.2711, 0.2532))
     ])
    
    transform_img_bsd100 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4248, 0.4339, 0.3699), (0.2445, 0.2322, 0.2374))
     ])

    transform_target_bsd100 = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize((0.4248, 0.4339, 0.3699), (0.2445, 0.2322, 0.2374))
     ])
    
    transform_img_urban100 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2629, 0.2552, 0.2757))
     ])

    transform_target_urban100 = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2629, 0.2552, 0.2757))
     ])
    
    transform_img_set14 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5315, 0.4494, 0.3984), (0.2638, 0.2607, 0.2524))
     ])

    transform_target_set14 = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize((0.5315, 0.4494, 0.3984), (0.2638, 0.2607, 0.2524))
     ])

    transform_img_div2k = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4287, 0.4300, 0.4020), (0.2671, 0.2569, 0.2866))
     ])

    transform_target_div2k = transforms.Compose(
        [transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize((0.4287, 0.4300, 0.4020), (0.2671, 0.2569, 0.2866))
     ])


    

    transform_img = [transform_img_set5, transform_img_bsd100, transform_img_urban100, transform_img_set14, transform_img_div2k]
    transform_target = [transform_target_set5, transform_target_bsd100, transform_target_urban100, transform_target_set14, transform_target_div2k]


elif resolution == '128/256':
    transform_img_set5 = transforms.Compose(
        [transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5075, 0.4057, 0.2967), (0.3092, 0.2655, 0.2480))
     ])

    transform_target_set5 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5075, 0.4057, 0.2967), (0.3092, 0.2655, 0.2480))
     ])
    
    transform_img_bsd100 = transforms.Compose(
         [transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.4249, 0.4339, 0.3699), (0.2390, 0.2266, 0.2324))
     ])

    transform_target_bsd100 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4249, 0.4339, 0.3699), (0.2390, 0.2266, 0.2324))
     ])
    
    transform_img_urban100 = transforms.Compose(
         [transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2501, 0.2427, 0.2643))
     ])

    transform_target_urban100 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2501, 0.2427, 0.2643))
     ])
    
    transform_img_set14 = transforms.Compose(
         [transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5314, 0.4493, 0.3983), (0.2573, 0.2536, 0.2456))
     ])

    transform_target_set14 = transforms.Compose(
        [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5314, 0.4493, 0.3983), (0.2573, 0.2536, 0.2456))
     ])

    transform_img_div2k = transforms.Compose(
         [transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.4154, 0.4181, 0.3953), (0.2603, 0.2495, 0.2768))
     ])

    transform_target_div2k = transforms.Compose(
       [transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.4154, 0.4181, 0.3953), (0.2603, 0.2495, 0.2768))
     ])


    

    transform_img = [transform_img_set5, transform_img_bsd100, transform_img_urban100, transform_img_set14, transform_img_div2k]
    transform_target = [transform_target_set5, transform_target_bsd100, transform_target_urban100, transform_target_set14, transform_target_div2k]

else:
    transform_img_set5 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5075, 0.4056, 0.2966), (0.3133, 0.2699, 0.2519))
     ])

    transform_target_set5 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5075, 0.4056, 0.2966), (0.3133, 0.2699, 0.2519))
     ])
    
    transform_img_bsd100 = transforms.Compose(
         [
            transforms.ToTensor(),
            transforms.Normalize((0.4248, 0.4338, 0.3699), (0.2454, 0.2331, 0.2381))
     ])

    transform_target_bsd100 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4248, 0.4338, 0.3699), (0.2454, 0.2331, 0.2381))
     ])
    
    transform_img_urban100 = transforms.Compose(
         [
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2757, 0.2680, 0.2872))
     ])

    transform_target_urban100 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4582, 0.4476, 0.4413), (0.2757, 0.2680, 0.2872))
     ])
    
    transform_img_set14 = transforms.Compose(
         [
            transforms.ToTensor(),
            transforms.Normalize((0.5315, 0.4494, 0.3984), (0.2638, 0.2607, 0.2524))
     ])

    transform_target_set14 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5315, 0.4494, 0.3984), (0.2638, 0.2607, 0.2524))
     ])

    transform_img_div2k = transforms.Compose(
         [
            transforms.ToTensor(),
            transforms.Normalize((0.4296, 0.4314, 0.4053), (0.2800, 0.2683, 0.2931))
     ])

    transform_target_div2k = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4296, 0.4314, 0.4053), (0.2800, 0.2683, 0.2931))
     ])
    

    transform_img = [transform_img_set5, transform_img_bsd100, transform_img_urban100, transform_img_set14, transform_img_div2k]
    transform_target = [transform_target_set5, transform_target_bsd100, transform_target_urban100, transform_target_set14, transform_target_div2k]




dataset_set5      = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
dataset_bsd100    = load_dataset('eugenesiow/BSD100', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
dataset_Urban100  = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
dataset_set14     = load_dataset('eugenesiow/Set14', 'bicubic_x2', split='validation', cache_dir = '/workspace/')
dataset_div2k     = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation', cache_dir = '/workspace/')


data = [dataset_set5, dataset_bsd100, dataset_Urban100, dataset_set14, dataset_div2k]
name = ["set5", "BSD100", "Urban100", "Set14", "Div2k"]

path =  'sisr_2x_128to256_bilinear_y_cbcrnoparam.pth'
model.load_state_dict(torch.load(path))

batch_size  = 1
ssim        = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr        = PeakSignalNoiseRatio().to(device)

model.eval()


for i, item in enumerate(data):

    valset     = SuperResolutionDataset(item, transform_img[i], transform_target[i])
    # valset     = SuperResolutionDataset(item, transform_img, transform_target)

    testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
    
    PSNR_rgb = 0
    SSIM_rgb = 0
    PSNR_ycbcr = 0
    SSIM_ycbcr = 0
    PSNR_y = 0
    SSIM_y = 0

    with torch.no_grad():
        for data in testloader:

            # Everything in ycbcr space
            images, labels = data[0].to(device), data[1].to(device) 
            outputs = model(images) 

            PSNR_ycbcr += psnr(outputs, labels) / len(testloader)
            SSIM_ycbcr += float(ssim(outputs, labels)) / len(testloader)

            # Extract Y Channel
            outputs_ychannel = outputs[:, 0:1, :, :]
            labels_ychannel = labels[:, 0:1, :, :]

            
            PSNR_y += psnr(outputs_ychannel, labels_ychannel) / len(testloader)
            SSIM_y += float(ssim(outputs_ychannel, labels_ychannel)) / len(testloader)

            #Convert to rgb space
            outputs = kornia.color.ycbcr_to_rgb(outputs)
            labels = kornia.color.ycbcr_to_rgb(labels)  
           
            PSNR_rgb += psnr(outputs, labels) / len(testloader)
            SSIM_rgb += float(ssim(outputs, labels)) / len(testloader)


    print(f"Dataset: {name[i]}\n")
    print("In RGB Space")
    print(f"PSNR: {PSNR_rgb:.2f} - SSIM: {SSIM_rgb:.4f}\n")
    print("In YCbCr Space")
    print(f"PSNR: {PSNR_ycbcr:.2f} - SSIM: {SSIM_ycbcr:.4f}\n")
    print("In Y Space")
    print(f"PSNR: {PSNR_y:.2f} - SSIM: {SSIM_y:.4f}\n")
