```text
.____                              ____   ____.__       .__                          
|    |    ____   _____   ____ _____\   \ /   /|__| _____|__| ____   ____       
|    |  _/ __ \ /     \ /    \\__  \\   Y   / |  |/  ___/  |/  _ \ /    \       
|    |__\  ___/|  Y Y  \   |  \/ __ \\     /  |  |\___ \|  (  <_> )   |  \    
|_______ \___  >__|_|  /___|  (____  /\___/   |__/____  >__|\____/|___|  /     
        \/   \/      \/     \/     \/                 \/               \/         
```
Welcome to LemnaVision! 

A light-weight convolutional neural network (CNN) for transforming images of duckweed into a binary maps, allowing you to calculate
duckweed area. Coded in python using pytorch.

The weights and model are provided for further personal training, although, the aim is to continuely increase model parameters
and training data to work for all duckweed species used for scientific experiments, and continuously update both here. Currently,
the model was trained on 92 manually segmented Lemna minuta images (well over 5000 individual fronds). Model inference will work
best on images taken ~30 cm above duckweed.

A frond is determined as an individual only when it has been seperated from the mother frond. This is to simplify inference.

A propper tutorial will be provided soon, and a script to allow for further training without the required knowledge of CNNs (for
those looking to use this for niche use cases).

For installation, install python requirments from the requirments.txt using either pip or conda. It is recomended to install using
a virtual env. To run, after activating your venv, run inference.py and follow the instructions given.

Model inference will currently only use the CPU, but only due to the version of pytorch specified in the requirements.txt (GPU is
supported in the code base). If your system has a GPU and you're fimilar with python, install a CUDA version of pytorch instead.
This will dramatically reduce inference time.

This is the first release so I'm looking for feedback, mostly from those familiar with python.
