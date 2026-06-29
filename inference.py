import cv2
import numpy as np
import pandas as pd
from src.cnn_tools import UNet, InferenceDataset, get_predictions
from src.img_tools import frond_counts, frond_area, avg_frond_area
from torch.utils.data import DataLoader
import torch
import os, tempfile
from pathlib import Path
import datetime
from art import *
from tqdm import tqdm
import builtins
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

# ---------------------------------------- #
#              user options                #
# ---------------------------------------- #

class userOptions():
    def __init__(self):
        # objects in calculation_type()
        self.cm_len = None

        # objects in batch_or_single()
        self.dir_or_file = None
        self.path = None

        self.calculation_type()
        self.batch_or_single()

    # check if the user wants to calculat area metrics
    def calculation_type(self):
        while True:
            cm_input = input('Do you want to calculate total duckweed area and average frond size? (y/n):').lower().strip()
            if cm_input == 'y':
                while True:
                    cm_len = input('Please input the number of pixels per centimetre (to the nearest int):')
                    try:
                        self.cm_len = int(cm_len)
                        break
                    except ValueError:
                        print('Invalid number. Please enter an integer.')
                        continue
                break
            elif cm_input == 'n':
                print('Only calculating semantic segmentation and fround counts.')
                self.cm_len = None
                break
            else:
                print('Please enter a valid input (int).')
                continue


    # check if user wants to process batch or single img
    def batch_or_single(self):
        while True:
            self.dir_or_file = input('Do you want to process a batch or single image (b/s)')

            # batch processing
            if self.dir_or_file == 'b':
                print('Batch processing selected.')
                # check if path exists
                while True:
                    self.path = input('Please input the directory containing images to process:')
                    if os.path.isdir(self.path):
                        print(f'Directory found at {self.path}')
                        break
                    else:
                        print(f'{self.path} is not a directory. Try again.')
                break

            # single file processing
            elif self.dir_or_file == 's':
                print('Single image processing selected.')
                # check if file exists
                while True:
                    self.path = input('Please input the path to your image:')
                    if os.path.isfile(self.path):                       
                        img = cv2.imread(self.path) # not loading img into an instance variable
                        if img is None:
                            print('Invalid file type. LemnaVision accepts .png .jpg .jpeg .tif .tiff. Try again.')
                            continue
                        else:
                            print(f'File found at {self.path}.')
                            break
                    else:
                        print(f'No file found at {self.path}. Please renter the path to your file.')
                break
            else:
                print('Incorrect input. Please pass "b" or "s"')

# ---------------------------------------- #
#             image calculations           #
# ---------------------------------------- #

class imageCalculations():
    def __init__(self, options):
        # taken from userOptions()
        self.cm_len = options.cm_len
        self.dir_or_file = options.dir_or_file
        self.path = options.path

        self.num_files = None
        self.valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.results_accumulator= []

        self.repeats()
        self.running()


    def repeats(self):
        if os.path.isfile(self.path):
            self.num_files = 1
        else:
            path = Path(self.path)
            self.num_files = sum(
                1 for x in path.iterdir()
                if x.is_file() and x.suffix.lower() in self.valid_ext
            )


    def running(self):
        # for single file
        if self.num_files == 1:
            img = cv2.imread(str(self.path))
            padded_img = self.pad_image(img)
            bmap_save_path = self.predict(padded_img, self.path)
            frond_num = self.frond_counting(bmap_save_path)
            if self.cm_len is not None:
                total_frond_area, avg_frond_area = self.calculate_area(bmap_save_path, self.cm_len, frond_num)
        
        # for batch analysis
        else:
            path = Path(self.path)

            files_to_process = [
                file for file in path.iterdir()
                if file.is_file() and file.suffix.lower() in self.valid_ext
            ]

            original_print = builtins.print
            builtins.print = tqdm.write
            
            try:
                for file in tqdm(files_to_process, desc='Processing Batch', leave=True):
                        img = cv2.imread(str(file))
                        padded_img = self.pad_image(img)
                        bmap_save_path = self.predict(padded_img, str(file.resolve()))
                        frond_num = self.frond_counting(bmap_save_path)
                        if self.cm_len is not None:
                            total_frond_area, avg_frond_area = self.calculate_area(bmap_save_path, self.cm_len, frond_num)
            finally:
                builtins.print = original_print

        # saving output pdf
        if self.results_accumulator and self.cm_len:
            outputs_dir = Path(bmap_save_path).parent.parent
            results_df = pd.DataFrame(self.results_accumulator)
            results_df.to_csv(f'{str(outputs_dir)}/results.csv')


    def pad_image(self, img, patch_size=256):
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


    def frond_counting(self, pred_path):
        print('Starting frond counting!')
        img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        frond_num, counted_img = frond_counts(img)
        tprint(f'{str(frond_num)}     fronds!')
        dirname = os.path.dirname(pred_path)
        file_name = os.path.splitext(os.path.basename(pred_path))[0]
        cv2.imwrite(f'{dirname}/{file_name}_counted.tif', counted_img)
        return frond_num
    
    def calculate_area(self, pred_path, cm_len, frond_num):
        # getting original image name
        splitted = os.path.splitext(os.path.basename(pred_path))[0].split('_')
        org_file_name = '_'.join(splitted[:-1])
        # load img and calculate area measurments
        img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        frond_space = round(frond_area(img, cm_len), 2)
        avg_frond = round(avg_frond_area(img, cm_len), 4)
        print(f'Area Calculations For Image: {org_file_name}')
        print(f'Total duckweed area: {frond_space} cm\u00b2')
        print(f'Average frond area: {avg_frond} cm\u00b2')
        # saving img and csv
        h, w = img.shape[:2]
        text = f'Fronds: {frond_num} -- Frond Coverage: {frond_space} cm2 -- Average Frond Area: {avg_frond} cm2'
        colour = (255, 255, 255)
        size = 5.0
        font = cv2.FONT_HERSHEY_PLAIN
        thickness = 10

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            fontScale=size,
            fontFace=font,
            thickness=thickness
            )
        x = (w - text_w) // 2
        y = h - 30

        cv2.putText(img,
                    text,
                    (x, y),
                    color=colour,
                    fontScale=size,
                    lineType=8,
                    fontFace=font,
                    thickness=thickness
                    )
        cv2.imwrite(f'{os.path.dirname(pred_path)}/{org_file_name}_area_details.tif', img)

        self.results_accumulator.append({
            'img': org_file_name,
            'frond_num': int(frond_num),
            'total_dw_area_cm2': float(frond_space),
            'avg_frond_area_cm2': float(avg_frond)
        })

        return frond_space, avg_frond


    def predict(self, padded_img, img_path, model=UNet(), patch_size=256):
        with tempfile.TemporaryDirectory() as tmpdir:
            print('Created temp directory. All temporary files will be removed after processing.')
            
            patch_size=patch_size
            num = 0
            img = padded_img
            img_name, ext = os.path.splitext(os.path.basename(img_path))
            os.makedirs(f'{os.path.dirname(os.path.abspath(img_path))}/outputs/{img_name}', exist_ok=True)
            save_path = f'{os.path.dirname(os.path.abspath(img_path))}/outputs/{img_name}'
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
            model.load_state_dict(torch.load('./weights/weights.pth', map_location=device))
            model.eval()
            model.to(device)

            with torch.no_grad():
                pred_paths = []
                for num, (batch_img, path) in enumerate(inference_loader):
                    path = path[0]
                    batch_img = batch_img.to(device)
                    seg, dist = model(batch_img)
                    p = get_predictions(seg, threshold=0.8)
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

        return f'{save_path}/{img_name}_predicted_bmap.tif'

options = userOptions()
imageCalculations(options)