```text
.____                              ____   ____.__       .__                          
|    |    ____   _____   ____ _____\   \ /   /|__| _____|__| ____   ____       
|    |  _/ __ \ /     \ /    \\__  \\   Y   / |  |/  ___/  |/  _ \ /    \       
|    |__\  ___/|  Y Y  \   |  \/ __ \\     /  |  |\___ \|  (  <_> )   |  \    
|_______ \___  >__|_|  /___|  (____  /\___/   |__/____  >__|\____/|___|  /     
        \/   \/      \/     \/     \/                 \/               \/         
```





<h4>About</h4>


LemnaVision is a light-weight convolutional neural network (CNN), utilisng a doubled-headed U-Net architecture for transforming images of duckweed into binary maps. These binary maps allow for accurate frond counting and area calculations, which are built into the program.

Model weights are provided, which can be personally trained for niche use cases. A training module is planned for future release to enable those unfamiliar with Python or PyTorch to train the model.

<h4>How it works</h4>

The model is trained on 92 manually segmented <i>Lemna minuta</i> plates. A frond is only defined as an individual if it is a seperate object. For example, a grouping of a mother, daughter and granddaughter frond, connected via their meristems will be counted as a single frond. This creates a standardised approach that removes the confusion of whether or not to count newly birthed tiny daughter fronds. This approach is common in the litterature. Furthermore, it simplies what the model needs to learn.

Ground truth binary maps are specifically segmented to emphasise gaps between individuals, including overlapping fronds. This is what allows post-process frond counting, which is done using connected-component analysis.

Total duckweed area is calculated based on the number of pixels in a cm. This needs to be calculated prior to running the model. I suggest including a ruler at media level in your image and calculate using imageJ. Briefly, select the line tool, draw a line on your ruler measuring 1 cm and press   ctrl+m. This will yield the number of pixels per cm, which is required as input to measure area.

Model inference supports both CPU and GPU. However, the requirements.txt specifies the CPU version of PyTorch. If you wish to speed up inference please install the appropriate version of PyTorch with CUDA support in your virtual environment: https://pytorch.org/get-started/locally/


<h4>Image requirements</h4>

Images should be on a white/light coloured background such as a piece of paper or light pad. A camera of at least 10 MP is required. A typical phone can produce images of sufficient resolution. Please use the test image (test.JPG) as an example.

<h4> Installation </h4>

For Those who are unfamilar with python, I'll be including the basics of creating a python virtual envionment and installing requirements. It isn't a comprehensive guide and if you have any issues I recomended reading the relavent docs.
Conda: https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
Python venv: https://docs.python.org/3/library/venv.html


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




