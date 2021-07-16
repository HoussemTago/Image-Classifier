from keras.models import load_model
import pickle
import numpy as np
with open('data.pickle','rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

model = load_model('image_classifier.h5')

print(model.summary())

