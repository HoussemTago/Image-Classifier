from keras.datasets import cifar10
import tensorflow.keras.utils as utils
import pickle

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

with open('data.pickle','wb') as f :
    pickle.dump([x_train, y_train, x_test, y_test],f)
