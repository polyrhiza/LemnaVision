import cv2
import numpy as np


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

    return frond_num

