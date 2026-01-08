import cv2
import numpy as np
from file_tools import load_paths, pad_images, patching, stratified_train_test_split
from cnn_tools import UNet, PatchDataset, precision, dice, get_predictions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


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
    batch_size=4,
    shuffle=True,
    num_workers=0
)

valLoader = DataLoader(
    valDataset,
    batch_size = 4,
    shuffle=False,
    num_workers=0
)

model = UNet()

device = torch.device('cpu')

model.to(device)

segCriterion = nn.BCEWithLogitsLoss() # Runs logits through sigmoid function then calculates binary cross entropy.
distCriterion = nn.L1Loss() # Measures mean absolute error.
distImportance = 0.3 # Relative importance of distCriterion.

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

numEpochs = 5

trainingData = {}

# ---------------------------------- #
# For reloading training checkpoints #
# ---------------------------------- #
checkpoint = torch.load('checkpoints/dw_seg_epoch_5.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
optimiser.load_state_dict(checkpoint['optimizer_state'])
startEpoch = checkpoint['epoch']  # Continue from next epoch
trainingData = checkpoint.get('trainingData', {})

# --------------------------- #
#   Train-Validation Loop
# --------------------------- #

for epoch in range(startEpoch+1, numEpochs+startEpoch+1):
    
    ##############
    #  TRAINING  #
    ##############

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

        # Forward #
        seg, dist = model(imgs) # Predictions

        # Loss #
        segLoss = segCriterion(seg, bmaps) # Loss calculation for segmentation.
        distLoss = distCriterion(dist, distMaps) # Loss calculation for distance maps.

        loss = segLoss + (distLoss * distImportance) # Final loss metric.

        # Back propagatrion #
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        trainLoss += loss.item()
        trainSegLoss += segLoss.item()
        trainDistLoss += distLoss.item()

        with torch.no_grad():

            for i in range(seg.shape[0]):
                img = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    trainPrecision += precision(img, bmap)
                    trainDice += dice(img, bmap)
                    numImgs += 1
    
            

    avgTrainLoss = trainLoss / len(trainLoader)
    avgTrainSegLoss = trainSegLoss / len(trainLoader)
    avgTrainDistLoss = trainDistLoss / len(trainLoader)

    if numImgs > 0:
        avgTrainPrecision = trainPrecision / numImgs
        avgTrainDice = trainDice / numImgs
    else:
        avgTrainPrecision = float('nan')
        avgTrainDice = float('nan') 

    ################
    #  VALIDATION  #
    ################

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

            # Forward #
            seg, dist = model(imgs) # Predictions

            # Loss #
            segLoss = segCriterion(seg, bmaps) # Loss calculation for segmentation.
            distLoss = distCriterion(dist, distMaps) # Loss calculation for distance maps.

            loss = segLoss + (distLoss * distImportance) # Final loss metric.

            valLoss += loss.item()
            valSegLoss += segLoss.item()
            valDistLoss += distLoss.item()

            for i in range(seg.shape[0]):
                img = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    valPrecision += precision(img, bmap)
                    valDice += dice(img, bmap)
                    valNumImgs += 1

    avgValLoss = valLoss / len(valLoader)
    avgValSegLoss = valSegLoss / len(valLoader)
    avgValDistLoss = valDistLoss / len(valLoader)

    if valNumImgs > 0:
        avgValPrecision = valPrecision / valNumImgs
        avgValDice = valDice / valNumImgs
    else:
        avgValPrecision = float('nan')
        avgValDice = float('nan')

    print(f'Epoch: [{epoch}/{numEpochs+startEpoch}]\n'
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
          f'Dice: {avgValDice:.4f}'
          f'############################')
    
    trainingData[f'Epoch{epoch+1}'] = {
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
        'trainingData': trainingData[f'Epoch{epoch+1}']},
        f'checkpoints/dw_seg_epoch_{epoch+1}.pth')



# ---------------- #
# Rerun validation #
# ---------------- #

epochs = 6 

for epoch in range(6, epochs+5):

    checkpoint = torch.load(f'checkpoints/dw_seg_epoch_{epoch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    trainingData = checkpoint.get('trainingData', {})
    model.eval()

    valPrecision = 0.0
    valDice = 0.0
    valNumImgs = 0.0

    with torch.no_grad():

        for imgs, bmaps, distMaps in valLoader:

            imgs = imgs.to(device)
            bmaps = bmaps.to(device)
            distMaps = distMaps.to(device)

            # Forward #
            seg, dist = model(imgs) # Predictions


            for i in range(seg.shape[0]):
                img = seg[i]
                bmap = bmaps[i]
                if bmap.sum() == 0:
                    continue
                else:
                    valPrecision += precision(img, bmap)
                    valDice += dice(img, bmap)
                    valNumImgs += 1


    if valNumImgs > 0:
        avgValPrecision = valPrecision / valNumImgs
        avgValDice = valDice / valNumImgs
    else:
        avgValPrecision = float('nan')
        avgValDice = float('nan')

    print('-----------------------------\n'
          f'Precision: {avgValPrecision:.4f}\n'
          f'Dice: {avgValDice:.4f}\n'
          f'----------------------------\n')
    
    trainingData['val_precision'] = avgValPrecision
    trainingData['val_dice'] = avgValDice

    torch.save(
        trainingData,
        f'checkpoints/dw_seg_epoch_{epoch}_reval.pth'
    )

# ------------------------------ #
#   Get predictions for viewing  #
# ------------------------------ #

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

epochs = [1, 5, 10]

for epoch in epochs:

    checkpoint = torch.load(f'checkpoints/dw_seg_epoch_{epoch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with torch.no_grad():

        for i, (img, bmap, distMap) in enumerate(testLoader):
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


