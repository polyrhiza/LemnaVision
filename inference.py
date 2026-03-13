import cv2
# import numpy as np
from src.cnn_tools import UNet, InferenceDataset, get_predictions
from src.img_tools import frond_counts
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
import os, tempfile

art = r"""                 
.____                              ____   ____.__       .__                          
|    |    ____   _____   ____ _____\   \ /   /|__| _____|__| ____   ____       
|    |  _/ __ \ /     \ /    \\__  \\   Y   / |  |/  ___/  |/  _ \ /    \       
|    |__\  ___/|  Y Y  \   |  \/ __ \\     /  |  |\___ \|  (  <_> )   |  \    
|_______ \___  >__|_|  /___|  (____  /\___/   |__/____  >__|\____/|___|  /     
        \/   \/      \/     \/     \/                 \/               \/         
                                                                        
"""                                                                                  

print(art)
print('Welcome to LemnaVision Inference Module!')

# ----------------------------------------------------------- #

def get_user_img():
    while True:
        img_path = input('Please input the path to your image:')

        if not os.path.isfile(img_path):
            print(f"File not found at '{img_path}'. Please check the path and try again.")
            continue
        
        img = cv2.imread(img_path)

        if img is None:
            print(f'File found at {img_path} but is either not an image or not a supported format.')
            continue

        print(f'Image of size {img.shape} detected. Moving to resizing.')
        return img, img_path

# ----------------------------------------------------------- #

def pad_img(img, patch_size=256):

    ''' Method for padding img for inference. Default is to make image divisable for 256.
        Added padding will be black bars on the top and right hand side of the img.

    Args: 
        img (array): img loaded as a numpy array.
        patch_size (int): img will be padded to be divisible by this size.

    Returns:
        padded_img (array): padded img divisible by given patch size.
    '''   
     
    h, w , c = img.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded_img = cv2.copyMakeBorder(
        img,
        top=pad_h,
        bottom=0,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )

    print(f"Padded image to size {padded_img.shape}.")

    return padded_img

# ----------------------------------------------------------- #

def predict(padded_img, img_path, model=UNet patch_size=256):
    ''' 
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Created temp directory.')

        patch_size=patch_size

        num = 0
        img = padded_img

        img_name, ext = os.path.splitext(os.path.basename(img_path))
        save_path = os.path.dirname(os.path.abspath(img_path))

        h, w , c = img.shape

        print('Patching image.')

        patched_paths = []
        coords = []

        for y in range (0, h, patch_size):
            for x in range(0, w, patch_size):
                img_patch = img[y: y+patch_size, x:x+patch_size]

                write_path = f'{tmpdir}/{img_name}_{num}.tif'

                cv2.imwrite(write_path, img_patch)
                
                patched_paths.append(write_path)

                coords.append((x, y))

                num += 1

        print(f'Patching complete. {num} patches created.')
        
        inference_set  = InferenceDataset(patched_paths)

        inference_loader = DataLoader(
            inference_set,
            batch_size=1,
            shuffle=False,
            num_workers = 0
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model
        model.load_state_dict(torch.load('./weights/LemnaVision_wights.pth'))
        model.eval()
        model.to(device)

        with torch.no_grad():
            for img, _ in inference_loader():
                img = img.to(device)
                seg, dist = model(img)
                preds = get_predictions(seg, threshold=0.8)

                for p in preds:
                    p = p.squeeze(0) * 255.0
            



        




