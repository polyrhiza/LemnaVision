import cv2
import numpy as np
from src.file_tools import load_paths, pad_images, patching, stratified_train_test_split
from src.cnn_tools import UNet, PatchDataset, precision, dice, get_predictions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import re
from natsort import natsorted



jpgPaths, bmapPaths = \
    load_paths('training_data/patches')

jpgTrain, bmapTrain, jpgVal, bmapVal = \
    stratified_train_test_split(jpgPaths,
                                bmapPaths)


trainDataset = PatchDataset(
    imgPaths=jpgTrain,
    bmapPaths=bmapTrain,
    augment=True
)

valDataset = PatchDataset(
    imgPaths=jpgVal,
    bmapPaths=bmapVal,
    augment=False
)

trainLoader = DataLoader(
    trainDataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

valLoader = DataLoader(
    valDataset,
    batch_size = 8,
    shuffle=False,
    num_workers=0
)

model = UNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

segCriterion = nn.BCEWithLogitsLoss() # Runs logits through sigmoid function then calculates binary cross entropy.
distCriterion = nn.L1Loss() # Measures mean absolute error.
distImportance = 0.4 # Relative importance of distCriterion.

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)


trainingData = {}

# ---------------------------------- #
# For reloading training checkpoints #
# ---------------------------------- #
checkpoint = torch.load('checkpoints/dw_seg_epoch_30.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
optimiser.load_state_dict(checkpoint['optimizer_state'])
trainingData = checkpoint.get('trainingData', {})

# --------------------------- #
#   Train-Validation Loop     #
# --------------------------- #

for epoch in range(31, 51):

    # ---------- #
    #  TRAINING  #
    # ---------- #

    model.train()

    trainLoss = 0.0
    trainSegLoss = 0.0
    trainDistLoss = 0.0

    trainPrecision = 0.0
    trainDice = 0.0
    numImgs = 0.0


    for imgs, bmaps, distMaps in trainLoader:

        imgs = imgs.to(device)
        bmaps = bmaps.to(device)
        distMaps = distMaps.to(device)

        # Forward 
        seg, dist = model(imgs) # Predictions

        # Loss 
        segLoss = segCriterion(seg, bmaps) # Loss calculation for segmentation.
        distLoss = distCriterion(dist, distMaps) # Loss calculation for distance maps.

        loss = segLoss + (distLoss * distImportance) # Final loss metric.

        # Back propagatrion 
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        trainLoss += loss.item()
        trainSegLoss += segLoss.item()
        trainDistLoss += distLoss.item()

        with torch.no_grad():

            for i in range(seg.shape[0]):
                segPred = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    trainPrecision += precision(segPred, bmap)
                    trainDice += dice(segPred, bmap)
                    numImgs += 1
    
            

    avgTrainLoss = trainLoss / len(trainLoader)
    avgTrainSegLoss = trainSegLoss / len(trainLoader)
    avgTrainDistLoss = trainDistLoss / len(trainLoader)

    if numImgs > 0:
        avgTrainPrecision = float(trainPrecision / numImgs)
        avgTrainDice = float(trainDice / numImgs)
    else:
        avgTrainPrecision = float('nan')
        avgTrainDice = float('nan') 

    # ------------ #
    #  VALIDATION  #
    # ------------ #

    model.eval()

    valLoss = 0.0
    valSegLoss = 0.0
    valDistLoss = 0.0

    valPrecision = 0.0
    valDice = 0.0
    valNumImgs = 0.0

    with torch.no_grad():

        for imgs, bmaps, distMaps in valLoader:

            imgs = imgs.to(device)
            bmaps = bmaps.to(device)
            distMaps = distMaps.to(device)

            # Forward 
            seg, dist = model(imgs) # Predictions

            # Loss 
            segLoss = segCriterion(seg, bmaps) # Loss calculation for segmentation.
            distLoss = distCriterion(dist, distMaps) # Loss calculation for distance maps.

            loss = segLoss + (distLoss * distImportance) # Final loss metric.

            valLoss += loss.item()
            valSegLoss += segLoss.item()
            valDistLoss += distLoss.item()

            for i in range(seg.shape[0]):
                segPred = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    valPrecision += precision(segPred, bmap)
                    valDice += dice(segPred, bmap)
                    valNumImgs += 1

    avgValLoss = valLoss / len(valLoader)
    avgValSegLoss = valSegLoss / len(valLoader)
    avgValDistLoss = valDistLoss / len(valLoader)

    if valNumImgs > 0:
        avgValPrecision = float(valPrecision / valNumImgs)
        avgValDice = float(valDice / valNumImgs)
    else:
        avgValPrecision = float('nan')
        avgValDice = float('nan')

    print(f'Epoch: [{epoch}/{30}]\n'
          f'Train Loss: {avgTrainLoss:.4f}\n'
          f'Seg: {avgTrainSegLoss:.4f}\n'
          f'Dist: {avgTrainDistLoss:.4f}\n'
          f'Precision: {avgTrainPrecision:.4f}\n'
          f'Dice: {avgTrainDice:.4f}\n'
          '-----------------------------\n'
          f'Val Loss: {avgValLoss:.4f}\n'
          f'Seg: {avgValSegLoss:.4f}\n'
          f'Dist: {avgValDistLoss:.4f}\n'
          f'Precision: {avgValPrecision:.4f}\n'
          f'Dice: {avgValDice:.4f}\n'
          f'############################\n')
    
    trainingData = {
        'train_loss': avgTrainLoss,
        'train_seg_loss': avgTrainSegLoss,
        'train_dist_loss': avgTrainDistLoss,
        'train_precision': avgTrainPrecision,
        'train_dice': avgTrainDice,
        'val_loss': avgValLoss,
        'val_seg_loss': avgValSegLoss,
        'val_dist_loss': avgValDistLoss,
        'val_precision': avgValPrecision,
        'val_dice': avgValDice
    }

    os.makedirs('checkpoints', exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimiser.state_dict(),
        'trainingData': trainingData},
        f'checkpoints/dw_seg_epoch_{epoch}.pth')

