```text
.____                              ____   ____.__       .__                          
|    |    ____   _____   ____ _____\   \ /   /|__| _____|__| ____   ____       
|    |  _/ __ \ /     \ /    \\__  \\   Y   / |  |/  ___/  |/  _ \ /    \       
|    |__\  ___/|  Y Y  \   |  \/ __ \\     /  |  |\___ \|  (  <_> )   |  \    
|_______ \___  >__|_|  /___|  (____  /\___/   |__/____  >__|\____/|___|  /     
        \/   \/      \/     \/     \/                 \/               \/         
```



<h4>About</h4>

LemnaVision is a light-weight convolutional neural network (CNN), utilisng a doubled-headed U-Net architecture, for transforming images of duckweed into semantically segmented binary maps. These binary maps allow for accurate frond counting and area calculations, which are built into the script.

Total duckweed area and average frond size is calculated based on a pixel-cm scale. This needs to be known prior to calculations. I suggest including a ruler at media level in your image and to calculate using imageJ.

Model inference supports both CPU and GPU. However, requirements.txt specifies the CPU version of PyTorch. If you wish to speed up inference please install the appropriate version of PyTorch with CUDA support in your virtual environment: https://pytorch.org/get-started/locally/.

For this purpose a frond is only defined as an individual if it is a distinct organism. For example, a grouping of mother, daughter and granddaughter frond connected via their meristems will be counted as a single frond. This is a standardised method common in the litterature.

Batch processing is also now supported.

<h4>Image requirements</h4>

Images should be on a white/light coloured background such as a piece of paper or light pad. A camera of at least 10 MP is required. A typical phone camera can produce images of sufficient resolution. Please use the test image (test.JPG) as an example.

`*.tiff`, `*.tif`, `*.jpg`, `*.jpeg'`, and `*.png` are supported file formats.

The binary maps will will saved in the folder that the original image was in.




