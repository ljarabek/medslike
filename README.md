# Detailed description: 
Inferring tumor location from body surface data.  https://drive.google.com/open?id=1njJyq18J6RnwVu2MXkT_tVk_jp5IWw5C

# How to use:
1. Run GetSurface.py and GetTumorCoordinates.py ; 
saves surface data and coordinate data in .npy files where .mha files are stored

2. Run model.py to train the model. The best are saved as Tensorflow checkpoint files.



*Known bug: GetTumorCoordinates.py - a subset of training set has glitched location - broken continuity = should be fixed!! --> method rework (Viola-Jones) or flood function
