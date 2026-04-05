import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2


##############################
#     IMAGE AUGMENTATION     #
##############################

def brightness_contrast_augmentation(img, p=0.5):
    if np.random.rand() > p:
        return img
    
    alpha = np.random.uniform(0.8, 1.2) # Adjusting contrast
    beta = np.random.uniform(-20, 20)

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

def gamma_augmentation(img, p=0.5):
    if np.random.rand() > p:
        return img
    
    gamma = np.random.uniform(0.8, 1.2)
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype('uint8')

    return cv2.LUT(img, table)

def blur_augmentation(img, p=0.5):
    if np.random.rand() > p:
        return img
    
    k = np.random.choice([3,5])
    return cv2.GaussianBlur(img, (k,k), 0)



def get_predictions(logits, threshold = 0.5, output='numpy'):
    '''
    Function that converts predictions to binary maps.

    Args:
        logits: Logit prediction returned from model().
        threshold: Probability threshold. Default is 0.5.
        output (numpy array, tensor): Dtype to return. Default is numpy.
    '''
    probs = torch.sigmoid(logits)
    preds = (probs > threshold)

    if output == 'numpy':
        return preds.cpu().numpy().astype('uint8')
    elif output == 'tensor':
        return preds.to(torch.uint8)
    else:
        raise ValueError('Unknown output type: {output}')
    

##############################
#           DATASETS         #
##############################

class PatchDataset(Dataset):

    def __init__(self, imgPaths, bmapPaths, augment=False):
        super().__init__()

        assert len(imgPaths) == len(bmapPaths)

        self.imgPaths = imgPaths
        self.bmapPaths = bmapPaths
        self.augment = augment

    def __len__(self):
        return len(self.imgPaths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imgPaths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bmap = cv2.imread(self.bmapPaths[index], cv2.IMREAD_GRAYSCALE)

        if img is None or bmap is None:
            raise RuntimeError(f'Failed to load data at index {index}')
        
        if self.augment:
            img = brightness_contrast_augmentation(img)
            img = blur_augmentation(img)
            img = gamma_augmentation(img)

        img = img.astype(np.float32) / 255.0 # Normalise.

        bmap = (bmap > 0 ).astype(np.float32) # Binarise bmaps.

        bmapUint8 = bmap.astype(np.uint8) # Required for opencv distance mask
        
        distMap = cv2.distanceTransform(
            1-bmapUint8, # Inverted as we want background distance to plant.
            cv2.DIST_L2, # Euclidean distance calculation.
            cv2.DIST_MASK_PRECISE # Exact euclidian distance. 
        )

        if distMap.max() > 0: # Check if there is distances.
            distMap = distMap / distMap.max() # Normalise distances around max dist.
        
        distMap = distMap.astype(np.float32)

        bmap = torch.from_numpy(bmap).unsqueeze(0) # Changes dimensions from H, W, C to C, H, W
        distMap = torch.from_numpy(distMap).unsqueeze(0)
        img = torch.from_numpy(img).permute(2, 0, 1) 

        return img, bmap, distMap
    

class InferenceDataset(Dataset):
    def __init__(self, imgPaths):
        super().__init__()

        self.imgPaths = imgPaths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self,index):
        img = cv2.imread(self.imgPaths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, self.imgPaths[index]

##############################
#          METRICS           #
##############################

def precision(logits, bmaps, threshold = 0.5):
    '''
    Function for calculating precision from passed logits.

                 true positive
        ------------------------------
        true positive + false positive
    
    logits: Single img as logits.
    bmaps: Single binary map.
    threshold: Probability threshold. Default is 0.5.
    '''
    epsilon =  1e-7 # Stops method returning undefined.

    probs = torch.sigmoid(logits) # Convert logits to probabilities.
    preds = (probs > threshold).float() # Create positive predictions based on given threshold. Convert float as bmaps are loaded as float32.

    truePositives = (preds * bmaps).sum()
    falsePositives = (preds * (1 - bmaps)).sum()
    precision = truePositives / (truePositives + falsePositives + epsilon)

    return precision

def dice(logits, bmaps, threshold = 0.5):
    '''
    Functions for calculating dice from passed logits.

                     2 * true positives   
    -----------------------------------------------------
     2 * true positves + false negatives + false positives


    logits: Single img as logits.
    bmaps: Single binary map.
    threshold: Probability threshold. Default is 0.5.
    '''
    epsilon = 1e-7 # Stops method returning undefined.

    probs = torch.sigmoid(logits) # Convert logits to probabilities.
    preds = (probs > threshold).float() # Create positive predictions based on given threshold. Convert float as bmaps are loaded as float32.

    truePositives = (preds * bmaps).sum()
    falsePositives = (preds * (1 - bmaps)).sum()
    falseNegatives = ((1 - preds) * bmaps).sum()
    dice = (2 * truePositives) / ((2 * truePositives) + falseNegatives + falsePositives + epsilon)

    return dice


##############################
#         CNN LAYERS         #
##############################


class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),

            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(inChannels, outChannels)

    def forward(self, x):
        x = self.pool(x)
        x=self.conv(x)

        return x


class Up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(inChannels, outChannels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x
    

class OutConv(nn.Module):
    def __init__(self, inChannels, outChannels=1):
        super().__init__()

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

##############################
#            U-NET           #
##############################
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.firstConv = DoubleConv(3, 64)
        self.firstDown = Down(64, 128)
        self.secondDown = Down(128, 256)

        self.firstUp = Up(256 + 128, 128)
        self.secondUp = Up(128 + 64, 64)

        self.segmentHead = OutConv(64, 1)
        self.distanceHead = OutConv(64, 1)

    def forward(self, x):
        x1 = self.firstConv(x)
        x2 = self.firstDown(x1)
        x3 = self.secondDown(x2)
        x4 = self.firstUp(x3, x2)
        x5 = self.secondUp(x4, x1)
        logits = self.segmentHead(x5)
        distances = self.distanceHead(x5)

        return logits, distances


    


