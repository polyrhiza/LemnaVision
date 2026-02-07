import cv2
import numpy as np
import os


def watershed(bmap):
    '''
    Method for watershedding binary maps to return untouched fronds.
    
    Arguments:
        bmap (numpy array): Binary map returned from CNN.

    Returns:
        ws_bmap (array, uint8): Newly watershed binary map.

    '''
    bmap = bmap.astype(np.uint) if bmap.dtype != np.uint8 else bmap
    bmap = (bmap*255).astype(np.uint8) if bmap.max() == 1 else bmap

    kernel = np.ones(
        (3,3),
        np.uint8
    )
    sure_bg = cv2.dilate(
        bmap,
        kernel,
        iterations=3
    )

    distmap = cv2.distanceTransform(
        bmap,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    _, sure_fg = cv2.threshold(
        distmap,
        0.3 * distmap.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown==255] = 0    

    labels = cv2.watershed(
        cv2.cvtColor(bmap, cv2.COLOR_GRAY2BGR),
        markers
    )

    ws_bmap = np.zeros(
        labels.shape,
        dtype=np.uint8
    )

    ws_bmap[labels>1] = 255

    return ws_bmap


def frond_counts(bmap):
    '''
    Counts the number of fronds without watershedding.
    
    Arguments
        bmap (numpy array): Binary map returned from CNN.

    Returns
        frond_num (int): Number of fronds found.
        bmap2 (numpy array): Array (x, y, 3) with fronds labeled.
    '''
    bmap = bmap.astype(np.uint) if bmap.dtype != np.uint8 else bmap
    bmap = (bmap*255).astype(np.uint8) if bmap.max() == 1 else bmap

    distmap = cv2.distanceTransform(
            bmap,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
    )

    _, sure_fg = cv2.threshold(
            distmap,
            0.3 * distmap.max(),
            255,
            0
    )

    sure_fg = np.uint8(sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    frond_num = int(np.max(np.unique(markers)))

    bmap2 = cv2.cvtColor(bmap, cv2.COLOR_GRAY2BGR)

    for frond in np.unique(markers):
        if frond < 1:
            continue

        frond_mask = np.uint8(markers == frond)

        M = cv2.moments(frond_mask)

        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00']) # 'm10 is the sum of x coordinates 'm00' is the total number of pixels.
            cy = int(M['m01'] / M['m00']) # 'm01' like above is the sum of all y coords.

            cv2.putText(
                bmap2,
                text=str(frond),
                org=(cX, cy),
                color=(200,0,0),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale=0.7,
                thickness=0
            )

    return frond_num, bmap2



def pad_images(jpgPaths, bmapPaths=None, savePath=None, patchSize=256):

    '''
    Method for padding training images from given paths and saving them out.
    
    jpgs: List of image arrays of size x, y, 3
    bmaps: List of binary map arrays of size x, y
    savePath
    patchsize: Size you want to patch. Default is 256.

    Returns: Paths to newly padded images and binary maps.
    '''   
    
    # assert len(jpgPaths) == len(bmapPaths)

    os.makedirs(savePath, exist_ok=True)

    paddedJpgPaths = []
    paddedBmapPaths = []

    for i in range(len(jpgPaths)):

        jpg = cv2.imread(jpgPaths[i])
        if bmapPaths != None:
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
        if bmapPaths != None:
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
        if bmapPaths != None:
            bmapName, ext = os.path.splitext(os.path.basename(bmapPaths[i]))

        jpgPath = f'{savePath}/{jpgName}_padded.tif'
        if bmapPaths != None:
            bmapPath = f'{savePath}/{bmapName}_padded.tif'

        cv2.imwrite(jpgPath, jpgPadded)
        if bmapPaths != None:
            cv2.imwrite(bmapPath, bmapPadded)

        paddedJpgPaths.append(jpgPath)
        if bmapPaths != None:
            paddedBmapPaths.append(bmapPath)

    if bmapPaths != None:
        return paddedJpgPaths, paddedBmapPaths
    else:
        return paddedJpgPaths



def patching(jpgPaths, savePath, bmapPaths=None, patchSize=256):

    '''
    Method for patching training data to desired size.
    
    jpgs: List of paths for jpg images
    bmaps: List of binary map paths
    patchsize: Size you want to patch. Default is 256.
    '''

    # assert len(jpgPaths) == len(bmapPaths)
    
    os.makedirs(savePath, exist_ok=True)

    jpgPatchPaths = []
    bmapPatchPaths = []
    coords = []
    

    for i in range(len(jpgPaths)):
        num = 0

        jpg = cv2.imread(jpgPaths[i])
        if bmapPaths != None:
            bmap = cv2.imread(bmapPaths[i], cv2.IMREAD_GRAYSCALE)

        jpgName, ext = os.path.splitext(os.path.basename(jpgPaths[i]))
        if bmapPaths != None:
            bmapName, ext = os.path.splitext(os.path.basename(bmapPaths[i]))

        h, w , c = jpg.shape

        for y in range (0, h, patchSize):
            for x in range(0, w, patchSize):
                jpgPatch = jpg[y: y+patchSize, x:x+patchSize]
                if bmapPaths != None:                
                    bmapPatch = bmap[y: y+patchSize, x:x+patchSize]

                jpgPath = f'{savePath}/{jpgName}_{num}.tif'
                if bmapPaths != None:
                    bmapPath = f'{savePath}/{bmapName}_{num}.tif'

                cv2.imwrite(jpgPath, jpgPatch)
                if bmapPaths != None:
                    cv2.imwrite(bmapPath, bmapPatch)
                
                jpgPatchPaths.append(jpgPath)
                if bmapPaths != None:
                    bmapPatchPaths.append(bmapPath)
                coords.append((x, y))

                num += 1

    if bmapPaths != None:
        return jpgPatchPaths, bmapPatchPaths, coords
    else:
        return jpgPatchPaths


