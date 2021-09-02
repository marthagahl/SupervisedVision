import os
import math
import numpy as np
import glob
import random
from PIL import Image
from tqdm import tqdm
import sys

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


#images_to_convert = glob.glob('/datasets01/imagenet_full_size/061417/train/n09*/*.JPEG', recursive = True)
images_to_convert = glob.glob('/checkpoint/mgahl/32_identities/*/*/*.jpg', recursive = True)
print (len(images_to_convert))

tf.reset_default_graph()

check_point = '/private/home/mgahl/DeepGazeII.ckpt'

if not os.path.exists('{}.meta'.format(check_point)):
    print ('Caught missing file')
    print (os.listdir('/private/home/mgahl/code'))
    sys.exit
try:
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
except Exception as e:
    print (e)


input_tensor = tf.get_collection('input_tensor')[0]
centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
log_density = tf.get_collection('log_density')[0]
log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    new_saver.restore(sess, check_point)

    for image in tqdm(sorted(images_to_convert)):
        name = image.split('/')
        
        ### Get original image
        im = Image.open(image).convert('RGB')

        img = np.asarray(im)

        ### Get input data
#        if im.mode != 'RGB':
#            print (im.mode)
#            print (name)
        image_data = img[np.newaxis, :, :, :]  #BHWC, three channels (RGB)
        centerbias_data = np.zeros((1, img.shape[0], img.shape[1], 1))  #BHWC, 1 channel (log density)

        ### Create tensorflow session, restore model parameters, compute log density prediction
        log_density_prediction = sess.run(log_density, {input_tensor: image_data, centerbias_tensor: centerbias_data})

        ### Get density prediction
        prob_arr = np.asarray(np.exp(log_density_prediction[0, :, :, 0]))

        im.close()

        if not os.path.exists('/checkpoint/mgahl/salience_maps_32ids/{}/{}/'.format(name[-3], name[-2])):
            os.makedirs('/checkpoint/mgahl/salience_maps_32ids/{}/{}/'.format(name[-3], name[-2]))

        np.save('/checkpoint/mgahl/salience_maps_32ids/{}/{}/{}.npy'.format(name[-3], name[-2], name[-1].split('.')[0]), prob_arr)

#        prob_img = Image.fromarray(prob_arr)
#        if prob_img.mode != 'RGB':
#            prob_img = prob_img.convert('RGB')
#        prob_img.save('/checkpoint/mgahl/salience_maps/{}'.format(name[-1]))
#


