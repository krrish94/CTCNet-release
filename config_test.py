import os
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

paths = dict(
    VGG11_PRETRAINED_WEIGHTS_PATH = "/home/ganeshiyer/vgg11_pretrained/vgg11_bn/vgg11_bn.pkl",
    DATASET_PATH = "/media/dataset/data/dataset/7_scenes_dataset/",
    PRETRAINED_WEIGHTS_PATH = "/home/ganeshiyer/VO-CTC/pretrained_weights/",
    # CHECKPOINT_PATH = "/tmp/ganesh_saved_Weights/ctc_long_windows/",
    CHECKPOINT_PATH = "/home/ganeshiyer/VO-CTC-KM/cache/lstm_ctc_long_windows/",
    checkpoint_no = 13
)

input_params = dict(
    IMG_HT = 240,
    IMG_WD = 240,
    model_input_mean = np.array([0.485, 0.456, 0.406]),
    model_input_std = np.array([0.229, 0.224, 0.225])
)

net_hyperparams = dict(
    sequence_length = 18,
    batch_size = 1,
    center_crop = True,
    isPartOfLSTM = True,
    end2end = True,
    n_hidden = 1000
	)
