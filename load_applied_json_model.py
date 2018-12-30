# -*- coding: utf-8 -*-
'''
Load a trainded keras model form json 
GHANMI Helmi

'''
from PIL import Image 
import numpy as np 
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

#---------------Load json model and create a new model based on weight of .h5 file ------------#

# load json and create model
json_file = open('trained_resnet50_model/custom_resnet50_model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("trained_resnet50_model/custom_resnet50_model2.h5")
print("Loaded model from disk")
#summary the model 
loaded_model.summary()

#---------------------test and predict result of an input image -------------#
#label in the trained model 
labels  =['humans','horses','dogs','cats']

#load and convert the input image into np array 
img_path='input_images_test/ronaldo.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

#preint the preduction (4 possibility as in labels list )
preds = loaded_model.predict(x)

# print the max of preduction and it labels
print('The input image is :',labels[np.argmax(preds)])
print('---------------')
print('max value of preduction is:',np.amax(preds))


