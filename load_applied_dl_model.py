# -*- coding: utf-8 -*-
'''
Load and make prediction using trainded keras model

GHANMI Helmi

'''


import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

#---------------Load model and create a new model based on weight of .h5 file ------------#
# load json and create model
json_file = open('trained_resnet50_model/custom_resnet50_model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("trained_resnet50_model/custom_resnet50_model2.h5")
print("Loaded model from disk")
# summary the model
loaded_model.summary()

#---------------------Make prediction-------------#
# label in the trained model
labels = ['humans', 'horses', 'dogs', 'cats']
img_path = 'input_images_test/ronaldo.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = loaded_model.predict(x)
print('Input image shape:', x.shape)
print('Prediction class:', labels[np.argmax(preds)])
print('Prediction score:', np.amax(preds))
