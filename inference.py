import cv2
import numpy as np
from src.cnn_tools import UNet, InferenceDataset, get_predictions
from src.img_tools import frond_counts, frond_counts_with_ws, frond_area
from torch.utils.data import DataLoader
import torch
import os, tempfile
from art import *
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

        print(f'Image of size {img.shape} detected.')

        while True:
            cm_input = input('Do you want to calculate duckweed area? (y/n):').lower().strip()

            if cm_input == 'y':

                while True:
                    cm_len = input('Please input number of pixels per centimetre (to the nearest int):')

                    try:
                        cm_len = int(cm_len)
                        break

                    except ValueError:
                        print('Invalid number. Please enter an integer.')
                        continue
                break

            elif cm_input == 'n':
                print('Only producing semantic segmentation and counting fronds.')
                cm_len = None
                break

            else:
                print('Please enter a valid input.')
                continue

        if cm_len:
            return img, img_path, int(cm_len)
        else:
            return img, img_path, None

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

def predict(padded_img, img_path, model=UNet(), patch_size=256):
    ''' 
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Created temp directory. All temporary files will be removed.')

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
                
                print(f'Temporary patch stored at {write_path}.')
        print('-----------------------------------------')
        print(f'Patching complete. {num} patches created.')
        print('Moving to Lemnaeae patch prediction.')
        print('-----------------------------------------')

        
        inference_set  = InferenceDataset(patched_paths)

        inference_loader = DataLoader(
            inference_set,
            batch_size=1,
            shuffle=False,
            num_workers = 0
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model
        model.load_state_dict(torch.load('./weights/weights.pth', map_location=device))
        model.eval()
        model.to(device)


        with torch.no_grad():
            pred_paths = []
            for num, (img, path) in enumerate(inference_loader):
                path = path[0]
            
                img = img.to(device)
                seg, dist = model(img)

                preds = get_predictions(seg, threshold=0.8)
                p = (preds.squeeze() * 255.0).astype(np.uint8)
                
                pred_path = os.path.dirname(path)
                pred_path = f'{pred_path}/{img_name}_pred_{num}.tif'
                pred_paths.append(pred_path)
                cv2.imwrite(pred_path, p)

                print(f'Temporary patch prediction saved at {pred_path} with unique values {np.unique(p)}.')
            
            print('---------------------------------')
            print('Finished predicting patches.')
            print('Stiching predicted patches.')
            print('---------------------------------')

        full_predict = np.zeros((h, w), dtype=np.uint8)

        for pred_path, patch_coords in zip(pred_paths, coords):
            print(f'pred path: {pred_path}')
            print(f'patch coords: {patch_coords}')
            print('---------------------------------')
            patch = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            x, y = patch_coords
            full_predict[y:y+patch_size, x:x+patch_size] = patch

        cv2.imwrite(f'{save_path}/{img_name}_predicted_bmap.tif', full_predict)
        print(f'Inference complete!')
        print(f'Lemnaceae binary map saved to {save_path}.')

    return f'{save_path}/{img_name}_predicted_bmap.tif', save_path, img_name



# ----------------------------------------------------------- #

def frond_counting(predicted_path, save_path, img_name):
    print('Starting frond counting!')
    img = cv2.imread(predicted_path, cv2.IMREAD_GRAYSCALE)
    frond_num, counted_img = frond_counts(img)
    tprint(f'{str(frond_num)}     fronds!')
    cv2.imwrite(f'{save_path}/{img_name}_counted.tif', counted_img)


# ----------------------------------------------------------- #

def calculate_area(img_path, cm_len):
    '''
    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    frond_space = frond_area(img, cm_len)

    print('Frond area:', round(frond_space, 2), 'cm\u00b2')
        
img, img_path, cm_len = get_user_img()
padded_img = pad_img(img)
predicted_path, save_path, img_name = predict(padded_img, img_path)
frond_counting(predicted_path, save_path, img_name)
if cm_len:
    calculate_area(predicted_path, cm_len)
else:
    pass

