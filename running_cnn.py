import cv2
import numpy as np
from src.file_tools import load_paths, pad_images, patching, stratified_train_test_split
from src.cnn_tools import UNet, PatchDataset, precision, dice, get_predictions
from src.img_tools import frond_counts
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import re
from natsort import natsorted
import yaml


jpgPaths, bmapPaths = \
    load_paths('training_data/originals')

paddedJpgs, paddedBmaps = \
    pad_images(jpgPaths=jpgPaths,
               bmapPaths=bmapPaths,
               savePath='training_data/padded')

patchedJpgs, patchedBmaps, patch_coords = \
    patching(jpgPaths=paddedJpgs,
             bmapPaths=paddedBmaps,
             savePath='training_data/patches')

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
distImportance = 0.8 # Relative importance of distCriterion. CHANGED AT 50

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

trainingData = {}

# ---------------------------------- #
# For reloading training checkpoints #
# ---------------------------------- #
checkpoint = torch.load('checkpoints/dw_seg_epoch_50.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
optimiser.load_state_dict(checkpoint['optimizer_state'])
trainingData = checkpoint.get('trainingData', {})

# --------------------------- #
#   Train-Validation Loop     #
# --------------------------- #

for epoch in range(51, 56):

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



# ---------------- #
# Rerun validation #
# ---------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.to(device)

for epoch in range(1, 11):
    print(f'Epoch: {epoch}')

    checkpoint = torch.load(f'./checkpoints/dw_seg_epoch_{epoch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    trainingData = checkpoint['trainingData']
    model.eval()

    trainPrecision = 0.0
    trainDice = 0.0
    trainNumImgs = 0.0

    with torch.no_grad():

        for imgs, bmaps, distMaps in trainLoader:

            imgs = imgs.to(device)
            bmaps = bmaps.to(device)
            distMaps = distMaps.to(device)

            # Forward #
            seg, dist = model(imgs) # Predictions


            for i in range(seg.shape[0]):
                segPred = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    trainPrecision += precision(segPred, bmap)
                    trainDice += dice(segPred, bmap)
                    trainNumImgs += 1


    if trainNumImgs > 0:
        avgTrainPrecision = float(trainPrecision / trainNumImgs)
        avgTrainDice = float(trainDice / trainNumImgs)
    else:
        avgTrainPrecision = float('nan')
        avgTrainDice = float('nan')

    print('-----------------------------\n'
          f'Precision: {avgTrainPrecision:.4f}\n'
          f'Dice: {avgTrainDice:.4f}\n'
          f'----------------------------\n')
    
    trainingData['train_precision'] = avgTrainPrecision
    trainingData['train_dice'] = avgTrainDice

    checkpoint['trainingData'] = trainingData
    
    torch.save(
        checkpoint,
        f'checkpoints/dw_seg_epoch_{epoch}.pth'
    )



# ------------------------------------------- #
#   Get predictions for viewing across epochs #
# ------------------------------------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.to(device)

imgPath = ['training_data/patches/DSC_0447_padded_129.tif',
           'training_data/patches/DSC_0433_padded_170.tif',
           'training_data/patches/DSC_0444_padded_107.tif']
bmapPath = ['training_data/patches/DSC_0447_BMAP_padded_170.tif',
            'training_data/patches/DSC_0433_BMAP_padded_170.tif',
            'training_data/patches/DSC_0444_BMAP_padded_107.tif']

testSet = PatchDataset(
    imgPath,
    bmapPath,
    augment=False
)

testLoader = DataLoader(
    testSet,
    batch_size = 1,
    shuffle=False,
    num_workers=0
)

preds = [[] for _ in range(3)]
imgs = [cv2.imread(img) for img in imgPath]

epochs = [1, 25, 55]

for epoch in epochs:

    checkpoint = torch.load(f'checkpoints/dw_seg_epoch_{epoch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with torch.no_grad():

        for i, (img, bmap, distMap) in enumerate(testLoader):
            img = img.to(device)
            seg, dist = model(img)
            pred = get_predictions(seg)
            preds[i].append(pred[0].squeeze(0))
            


fig, ax = plt.subplots(3, 4, figsize=(10,10))

for row in range(0,3):
    ax[row,0].imshow(imgs[row])

    for col in range(1, 4):
        ax[row, col].imshow(preds[row][col-1], cmap='gray')
        ax[row, col].set_title(f"Epoch {epochs[col-1]}")
        ax[row, col].axis('off')

plt.tight_layout()
plt.show()


# ------------------------------ #
#     Graph loss and metrics     #
# ------------------------------ #


fileList = []
for root, dir, file in os.walk('./checkpoints/'):
    for i in file:
        fileList.append(os.path.join(root, i))

fileList = natsorted(fileList)


trainPrecision = []
trainDice = []
trainSegLoss = []
trainDistLoss = []

valPrecision = []
valDice = []
valSegLoss = []
valDistLoss = []

for i in fileList:
    checkpoint = torch.load(i)
    trainingData = checkpoint['trainingData']
    trainSegLoss.append(trainingData['train_seg_loss'])
    trainDistLoss.append(trainingData['train_dist_loss'])
    trainDice.append(trainingData['train_dice'])
    trainPrecision.append(trainingData['train_precision'])

    valSegLoss.append(trainingData['val_seg_loss'])
    valDistLoss.append(trainingData['val_dist_loss'])
    valPrecision.append(trainingData['val_precision'])
    valDice.append(trainingData['val_dice'])
        
fig, ax = plt.subplots(1, 2)

ax[0].plot(range(1, len(fileList)+1), trainSegLoss)
ax[0].plot(range(1, len(fileList)+1), trainDistLoss)
# ax[0].plot(range(1, len(fileList)+1), trainDice)
# ax[0].plot(range(1, len(fileList)+1), trainPrecision)

ax[1].plot(range(1, len(fileList)+1), valSegLoss)
ax[1].plot(range(1, len(fileList)+1), valDistLoss)
# ax[1].plot(range(1, len(fileList)+1), valDice)
# ax[1].plot(range(1, len(fileList)+1), valPrecision)

plt.plot()



# -------------------------------------- #
#    Patch and predict then reconnect    #
# -------------------------------------- #


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.to(device)

imgPath = ['./training_data/padded/DSC_0352_padded.tif']
bmapPath = ['./training_data/padded/DSC_0352_BMAP_padded.tif']
savePath = './testing'

jpgPatch, bmapPatch, coords = patching(imgPath, bmapPath,savePath)

testSet = PatchDataset(
    jpgPatch,
    bmapPatch,
    augment=False
)

testLoader = DataLoader(
    testSet,
    batch_size = 1,
    shuffle=False,
    num_workers=0
)

epoch = 55

checkpoint = torch.load(f'checkpoints/dw_seg_epoch_{epoch}.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

preds = []

with torch.no_grad():

    for i, (img, bmap, distMap) in enumerate(testLoader):

        img = img.to(device)
        seg, dist = model(img)
        pred = get_predictions(seg)
        preds.append(pred[0].squeeze(0))
        

cols = 20
rows = []

for i in range(0, len(preds), cols):
    row = np.concatenate(preds[i:i+cols], axis=1)
    rows.append(row)

final_img = np.concatenate(rows, axis=0)


frond_num, counted_img = frond_counts(final_img)
