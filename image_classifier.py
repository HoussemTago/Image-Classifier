from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.constraints import maxnorm
import pickle

with open('data.pickle','rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)


model = Sequential()
conv_layer = Conv2D(filters=32,kernel_size=(3,3),input_shape = (32,32,3),activation='relu',padding ='same',
                    kernel_constraint=maxnorm(3))
max_pool_layer = MaxPooling2D(pool_size=(2,2))
dense_layer1 = Dense(units=1024, activation='relu',kernel_constraint=maxnorm(3))
dropout_layer = Dropout (rate = 0.5)
dense_layer2 = Dense(units=10, activation='softmax')

model.add(conv_layer)
model.add(max_pool_layer)
model.add(Flatten())
model.add(dense_layer1)
model.add(dropout_layer)
model.add(dense_layer2)


model.compile(optimizer = SGD(learning_rate =0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=60,batch_size=32)
model.save(filepath='image_classifier.h5')


