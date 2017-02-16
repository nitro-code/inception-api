import tensorflow as tf
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np
import h5py


model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
graph = tf.get_default_graph()

def pil2array(pillow_img):
    return np.array(pillow_img.getdata(), np.float32).reshape(pillow_img.size[1], pillow_img.size[0], 3)

def predict_pil(pillow_img):
    img_array = pil2array(pillow_img)
    return predict_nparray(img_array)

def predict_nparray(img_as_array):
    global graph

    img_as_array = np.expand_dims(img_as_array, axis=0)
    img_as_array = preprocess_input(img_as_array)

    with graph.as_default():
        preds = model.predict(img_as_array)

    decoded_preds = decode_predictions(preds, top=3)[0]
    predictions = [{'label': label, 'descr': description, 'prob': probability} for label,description, probability in decoded_preds]
    return predictions
