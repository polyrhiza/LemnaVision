```text
.____                              ____   ____.__       .__                          
|    |    ____   _____   ____ _____\   \ /   /|__| _____|__| ____   ____       
|    |  _/ __ \ /     \ /    \\__  \\   Y   / |  |/  ___/  |/  _ \ /    \       
|    |__\  ___/|  Y Y  \   |  \/ __ \\     /  |  |\___ \|  (  <_> )   |  \    
|_______ \___  >__|_|  /___|  (____  /\___/   |__/____  >__|\____/|___|  /     
        \/   \/      \/     \/     \/                 \/               \/         
```





<h4>About</h4>


LemnaVision is a light-weight convolutional neural network (CNN), utilisng a reduced U-Net architecture for transforming images of duckweed into binary maps. Forther allowing frond counts and area calculations.

A frond is determined as an individual only when it has been seperated from the mother frond. This is to standardise counting and simply model inference.

Model inference is set up to use the CPU, but only due to the version of pytorch specified in the requirements.txt (GPU is supported in the code base). If your system has a GPU and you're fimilar with python, install a CUDA version of pytorch instead. This will dramatically reduce inference time.

The weights and model are provided for further personal training for niche use cases, although, the aim is to continuely training data to work for all duckweed species used for scientific experiments, and continuously update both here. Currently, the model was trained on 92 manually segmented <i>Lemna minuta</i> images (well over 5000 individual fronds). Model inference will work best on images taken approximately 10 < x < 30 cm above duckweed, with a light background.

This is the first release so I'm looking for feedback! 
If you get any issues please report them in the issues tab.


<h4> Installation </h4>

For Those who are unfamilar with python, I'll be including the basics of creating a python virtual envionment and installing requirements.

<h5>1. Clone repository: </h5>

Download as a zip or use:
`git clone https://github.com/polyrhiza/LemnaVision` 

<h5>2. Create and activate a virtual environment:</h5>

<h6>Using python venv:</h6>

`python -m venv LemnaVision python=3.14`

To activate on mac/linux use:
`source ./LemnaVision/bin/activate`

or on windows:
`LemnaVision/Scripts/activate`

<h6>Using conda env:</h6>

`conda create -n LemnaVision python=3.14`
`conda activate LemnaVision`


<h5>3. Install requirements:</h5>

Navigate to the cloned repo and use:
`pip install -r requirements.txt`

<h5>4. Run</h5>

While in the repo directory run:
`python inference.py`

You will be prompted to add the path to the image you want to pass through the model.

`*.tiff`, `*.tif`, `*.jpg`, `*.jpeg'`, and `*.png` are supported file formats.

The output images will saved in the folder that the original image was in.




