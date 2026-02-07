import os
import numpy as np
import re
import cv2
import torchvision as T

folder = 'training_data/originals'

def load_paths(folder):
    '''
    Method for getting paths top matching jpgs and binary maps.
    
    folder: required to be a folder with folders within, each folder
            containing images and binary maps within.

    returns: bmaps = list of binary maps in grayscale as numpy arrays.
             jpgs = list of jpgs in RGB as numpy arrays.
    '''

    jpgPaths = []
    bmapPaths = []
    for i in os.listdir(folder):

        
        path = os.path.join(folder, i)

        if os.path.isdir(path): # Check if what is inside the given folder is another folder.
            for j in os.listdir(path):
                if re.search(r'BMAP', j) and os.path.isfile(os.path.join(path, j)): 
                    bmapPaths.append(os.path.join(path, j))
                elif os.path.isfile(os.path.join(path, j)) and not re.search(r'BMAP', j) and j.endswith('.JPG'):
                    jpgPaths.append(os.path.join(path, j))
        
        if os.path.isfile(path): # Check what is inside given folder is a file.
            if re.search(r'BMAP', i):
                bmapPaths.append(path)
            elif os.path.isfile(path) and not re.search(r'BMAP', i):
                jpgPaths.append(path)

    if len(bmapPaths) != 0:
        jpgNums = []
        for jpg in jpgPaths:
            file = os.path.basename(jpg)
            jpgNum = re.search(r'\d+', file).group()
            jpgNums.append(jpgNum)

        bmapNums = []
        for bmap in bmapPaths:
            file = os.path.basename(bmap)
            bmapNum = re.search(r'\d+', file).group()
            bmapNums.append(bmapNum)

        toKeep = [x for x in jpgNums if x in bmapNums]

        jpgPaths = sorted([x for x in jpgPaths if re.search(r'\d+', os.path.basename(x)).group()in toKeep])
        bmapPaths = sorted(bmapPaths)

    if len(bmapPaths) == len(jpgPaths):
        return jpgPaths, bmapPaths
    else:
        return jpgPaths


def stratified_train_test_split(jpgPaths, bmapPaths, split = 0.8, seed=42):
    """
    Method for giving a stratified train-test split.

    jpgPaths: List of RGB patch paths.
    bmapPaths: List of binary map patch paths.
    split: Proportion of patches in training data.
    seed: Random seed for shuffling training data.

    Returns:
        
    """  

    assert len(jpgPaths) == len(bmapPaths)
    
    sumArray = []
    
    for i in range(len(bmapPaths)):
        bmap = cv2.imread(bmapPaths[i], cv2.IMREAD_GRAYSCALE)
        sum = bmap.sum()
        sumArray.append(0 if sum <= 0 else 1)

    bmapPaths = np.asarray(bmapPaths)
    jpgPaths = np.asarray(jpgPaths)
    sumArray = np.asarray(sumArray)

    plantMask = sumArray.astype(bool)
    
    bmapPlants = bmapPaths[plantMask]
    bmapBackground = bmapPaths[~plantMask]
    jpgPlants = jpgPaths[plantMask]
    jpgBackground = jpgPaths[~plantMask]

    plantIdx = int(np.round(len(bmapPlants) * split))
    bgdIdx = int(np.round(len(bmapBackground) * split))

    bmapTrain = np.concat([bmapPlants[:plantIdx],
                       bmapBackground[:bgdIdx]])
    
    bmapTest = np.concat([bmapPlants[plantIdx:],
                      bmapBackground[bgdIdx:]])

    jpgTrain = np.concat([jpgPlants[:plantIdx],
                          jpgBackground[:bgdIdx]])
    
    jpgTest = np.concat([jpgPlants[plantIdx:],
                         jpgBackground[bgdIdx:]])
    
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(jpgTrain))

    jpgTrain = jpgTrain[perm]
    bmapTrain = bmapTrain[perm]
    
    return jpgTrain.tolist(), bmapTrain.tolist(), jpgTest.tolist(), bmapTest.tolist()
    











    













