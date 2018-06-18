import numpy as np
import tensorflow as tf

import argparse
import glob
from natsort import natsorted as ns
import scipy.misc as smc
from skimage.transform import resize
from tqdm import tqdm
import sys
sys.dont_write_bytecode = True

from models import models_cnn
from models.models_lstm_mult import dynamic_RNN
from models.cnn_utils import weight_variable
from models.liefunctions import *



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'Name of the config file (exclude the .py suffix)')
    args = parser.parse_args()
    config = __import__(args.config)

    # Input image dimensions and parameters
    IMG_HT = config.input_params['IMG_HT']
    IMG_WD = config.input_params['IMG_WD']
    norm_mean = config.input_params["model_input_mean"]
    norm_std = config.input_params["model_input_std"]

    # Test Network hyperparameters
    batch_size = config.net_hyperparams['batch_size']
    seq_len = config.net_hyperparams['sequence_length']

    # LSTM hyperparmaeters
    if(config.net_hyperparams['center_crop'] == True):
        n_input = 1024
    else:
        n_input = 1280
    n_hidden = config.net_hyperparams['n_hidden']
    W_rnn = weight_variable([n_hidden, 6], 13, decay = 0.001)

    # Create placeholder for input data
    X = tf.placeholder(tf.float32, shape = (seq_len, IMG_HT, IMG_WD, 6))

    # Phase (train vs test) indicator
    phase = tf.placeholder(tf.bool, [])

    # Droupout retain probabilities
    lstm_keep_prob = tf.placeholder(tf.float32, shape = ())
    keep_prob = tf.placeholder(tf.float32, shape = ())
    
    # CNN model
    Model_obj = models_cnn.CNN_VGG11_BN(X, phase, keep_prob, seq_len, \
		config.paths['VGG11_PRETRAINED_WEIGHTS_PATH'], isPartOfLSTM = True, retrainCNN = False, \
		end2end = config.net_hyperparams['end2end'], centerCrop = True)
    
    output, var_summaries = Model_obj.inference()

    lstm_input = tf.reshape(output, (1, seq_len, output.get_shape().as_list()[1]))
    
    composite_vectors_input = lstm_input[:,:seq_len,:]
    
    composite_xi = dynamic_RNN(composite_vectors_input[0], W_rnn, seq_len, n_input, n_hidden, reuse = False, dropout = lstm_keep_prob)

    saver = tf.train.Saver()
    
    sequence_path = config.paths["DATASET_PATH"]
    
    dict_test_split = {"chess":["03","05"], "fire":["03","04"], "heads":["01"], "office": ["02","06","07","09"], "pumpkin":["01","07"], "redkitchen":["03","04","06","12"]}
    
    # GPU Configuration
    config_tf = tf.ConfigProto(allow_soft_placement=True)
    config_tf.gpu_options.allow_growth = True

    with tf.Session(config = config_tf) as sess:
        sess.run(tf.global_variables_initializer())

        print("Restoring Checkpoint")

        ckpt_no = config.paths['checkpoint_no']
        checkpoints_file_name = config.paths["CHECKPOINT_PATH"] + "model-%d"%ckpt_no
        saver.restore(sess, checkpoints_file_name)

        for subfolder,seq_list in dict_test_split.iteritems():

            for seq_no in seq_list:

                print("Trajectory_output_len_" + str(seq_len) + "_" + subfolder + "_" + seq_no + "_" + str(ckpt_no) + ".txt")

                images_path = sequence_path + subfolder + "/seq-" + seq_no + "/color/*.png"
                image_file_names = ns(glob.glob(images_path))

                gt_poses_path = sequence_path + subfolder + "/seq-" + seq_no + "/pose/*.txt"
                gt_pose_file_names = ns(glob.glob(gt_poses_path))


                plot_point_set = np.zeros((len(image_file_names), 4))
                plot_point_set[0,:] = np.array([0.0, 0.0, 0.0, 1.0])

                xi_pred = np.zeros((len(image_file_names), 6))

                gt_point_set = np.zeros((len(image_file_names), 4))
                gt_point_set[0,:] = np.array([0.0, 0.0, 0.0, 1.0])

                xi_gt = np.zeros((len(image_file_names), 6))

                old_xi = np.zeros((seq_len, 6))

                for iter_id in tqdm(range(0, len(image_file_names) - seq_len, 1)):

                    ip_container = np.zeros((seq_len, IMG_HT, IMG_WD, 6))
                    for seq_id in range(seq_len):
                        source_name = image_file_names[iter_id+seq_id]
                        target_name = image_file_names[iter_id + seq_id + 1]

                        source_img = smc.imread(source_name)
                        target_img = smc.imread(target_name)

                        ht = source_img.shape[0]
                        wdt = source_img.shape[1]
                        lim1 = (wdt - ht)/2
                        lim2 = wdt - lim1

                        source_img = np.float32(source_img)
                        source_img = source_img[:,lim1:lim2]

                        source_img = resize(source_img, (IMG_HT, IMG_WD), preserve_range=True)
                        source_img = (source_img)/255.0
                        source_img = (source_img -  norm_mean)/norm_std

                        target_img = np.float32(target_img)
                        target_img = target_img[:,lim1:lim2]

                        target_img = resize(target_img, (IMG_HT, IMG_WD), preserve_range=True)

                        target_img = (target_img)/255.0
                        target_img = (target_img -  norm_mean)/norm_std

                        net_ip = np.concatenate((source_img, target_img), 2)

                        ip_container[seq_id] = net_ip

                    xi_out = sess.run(composite_xi, feed_dict={X:ip_container, phase:False, keep_prob:1.0, lstm_keep_prob:1.0})
                    

                    ## Windowd averaging, use the sequence_length to extract window and add, and average

                    if(iter_id  == 0):
                        old_xi = xi_out

                    elif(iter_id > 0 ):

                        if iter_id < seq_len:
                            mult = iter_id + 1
                        else:
                            mult = seq_len

                        xi_prev = (mult - 1)*old_xi[1:]
                        averaged = (xi_prev + xi_out[:-1])/mult
                        xi_out = np.vstack((averaged, xi_out[-1]))
                        old_xi = xi_out


                    for seq_id in range(seq_len):

                        xi_pred[iter_id + seq_id] = xi_out[seq_id]

                        local_transform =  SE3_expmap(xi_out[seq_id])

                        old_point = plot_point_set[iter_id + seq_id].reshape(4,1)

                        plot_point_set[iter_id + seq_id + 1] = np.matmul(local_transform, old_point)[:,0]

                        global_pose1 = np.loadtxt(gt_pose_file_names[iter_id + seq_id], dtype = np.float32)
                        global_pose2 = np.loadtxt(gt_pose_file_names[iter_id + seq_id + 1], dtype = np.float32)

                        gt_local_transform = np.matmul(global_pose2, np.linalg.inv(global_pose1))
                        xi_gt_local = SE3_logmap(gt_local_transform)
                        xi_gt[iter_id + seq_id] = xi_gt_local

                        gt_old_point = gt_point_set[iter_id + seq_id]
                        gt_point_set[iter_id + seq_id + 1] = np.matmul(gt_local_transform, gt_old_point)

                save_arr = np.hstack((plot_point_set[:,:3], xi_pred, gt_point_set[:,:3], xi_gt))

                np.savetxt("./Trajectory_output_len_" + str(seq_len) + "_" + subfolder + "_" + seq_no + "_" + str(ckpt_no) + ".txt", save_arr)
