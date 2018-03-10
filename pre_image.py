import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

img = misc.imread("/home/z/PycharmProjects/detectface/database/0.jpg")
with tf.Session() as sess:

    img0 = tf.image.per_image_standardization(img)
    plt.imshow(img)
    plt.show()