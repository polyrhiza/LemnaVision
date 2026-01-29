import os
import numpy as np
import re
import cv2
import torchvision as T

# folder = 'training_data/originals'

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

    return jpgPaths, bmapPaths



def pad_images(jpgPaths, bmapPaths, savePath, patchSize=256):

    '''
    Method for padding training images from given paths and saving them out.
    
    jpgs: List of image arrays of size x, y, 3
    bmaps: List of binary map arrays of size x, y
    savePath
    patchsize: Size you want to patch. Default is 256.

    Returns: Paths to newly padded images and binary maps.
    '''   
    
    assert len(jpgPaths) == len(bmapPaths)

    os.makedirs(savePath, exist_ok=True)

    paddedJpgPaths = []
    paddedBmapPaths = []

    for i in range(len(jpgPaths)):

        jpg = cv2.imread(jpgPaths[i])
        bmap = cv2.imread(bmapPaths[i], cv2.IMREAD_GRAYSCALE)
        
        h, w , c = jpg.shape
        pad_h = (patchSize - h % patchSize) % patchSize
        pad_w = (patchSize - w % patchSize) % patchSize

        jpgPadded = cv2.copyMakeBorder(
            jpg,
            top=pad_h,
            bottom=0,
            left=0,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0,0,0)
        )

        bmapPadded = cv2.copyMakeBorder(
            bmap,
            top=pad_h,
            bottom=0,
            left=0,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=0
            )
        
        

        jpgName, ext = os.path.splitext(os.path.basename(jpgPaths[i]))
        bmapName, ext = os.path.splitext(os.path.basename(bmapPaths[i]))

        jpgPath = f'{savePath}/{jpgName}_padded.tif'
        bmapPath = f'{savePath}/{bmapName}_padded.tif'

        cv2.imwrite(jpgPath, jpgPadded)
        cv2.imwrite(bmapPath, bmapPadded)

        paddedJpgPaths.append(jpgPath)
        paddedBmapPaths.append(bmapPath)


    return paddedJpgPaths, paddedBmapPaths



def patching(jpgPaths, bmapPaths, savePath, patchSize=256):

    '''
    Method for patching training data to desired size.
    
    jpgs: List of paths for jpg images
    bmaps: List of binary map paths
    patchsize: Size you want to patch. Default is 256.
    '''

    assert len(jpgPaths) == len(bmapPaths)
    
    os.makedirs(savePath, exist_ok=True)

    jpgPatchPaths = []
    bmapPatchPaths = []
    coords = []
    

    for i in range(len(jpgPaths)):
        num = 0

        jpg = cv2.imread(jpgPaths[i])
        bmap = cv2.imread(bmapPaths[i], cv2.IMREAD_GRAYSCALE)

        jpgName, ext = os.path.splitext(os.path.basename(jpgPaths[i]))
        bmapName, ext = os.path.splitext(os.path.basename(bmapPaths[i]))

        h, w , c = jpg.shape

        for y in range (0, h, patchSize):
            for x in range(0, w, patchSize):
                jpgPatch = jpg[y: y+patchSize, x:x+patchSize]
                bmapPatch = bmap[y: y+patchSize, x:x+patchSize]

                jpgPath = f'{savePath}/{jpgName}_{num}.tif'
                bmapPath = f'{savePath}/{bmapName}_{num}.tif'

                cv2.imwrite(jpgPath, jpgPatch)
                cv2.imwrite(bmapPath, bmapPatch)
                
                jpgPatchPaths.append(jpgPath)
                bmapPatchPaths.append(bmapPath)
                coords.append((x, y))

                num += 1
    
    return jpgPatchPaths, bmapPatchPaths, coords



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
    











    













