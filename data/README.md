## Data Preparation

#### RGB Frames
CTCNet was trained and tested on RGB frame sequences from the [Microsoft 7-Scenes RGB-D dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/). We use all the sequences from the dataset. We also follow the train/test splits as mentioned.

#### ORB-SLAM data-preparation
To facilitate training CTCNet, we first collect ORB-SLAM estimates for the training sequences. Since we need to maintain scale between sequences, we use the [RGB-D pipeline](https://github.com/raulmur/ORB_SLAM2#6-rgb-d-example) of the [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) package.
We make the following modifications to the RGB-D example:
* In order to treat it as a visual odometry pipeline, we disable Loop Closure and Relocalization.
* In order to check for consistency between transformations at different frame intervals (t, t+1, t+2), we collect per frame local pose transformations at different frame rates, and provide the sequences at different intervals. This results in the follwing sets of sequences for each sequence in the dataset:
    * interval t: (frame 0, frame 1, frame 2 ....)
    * interval t+1: (frame 0, frame 2, frame 4 ....)
    * interval t+1: (frame 1, frame 3, frame 5 ....)
    * interval t+2: (frame 0, frame 3, frame 6 ....)
    * interval t+2: (frame 1, frame 4, frame 7 ....)
    * interval t+2: (frame 2, frame 5, frame 8 ....)

These sets can now be used for composing longer windows for training the LSTM. As a default, we compose longer windows of size 18 during training. For the sake of brevity, we provide our parsing files (`train_parser.txt` and `test_parser.txt`) and per sequence transforms. 
