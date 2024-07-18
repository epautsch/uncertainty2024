import argparse
import keras
import tensorflow as tf
from keras.datasets import mnist
#from keras.utils import np_utils
from keras.utils import to_categorical
#from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Average
from keras.layers import Lambda
from tensorflow.keras import optimizers
from keras import backend as K
import time


parser = argparse.ArgumentParser(description='Fusion Ensembles NN')
parser.add_argument('--gpus', type=int, default=1, help='num gpus to use')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--num_ensembles', type=int, default=3, help='numba of epochs')
args = parser.parse_args()

gpus = args.gpus
batch_size = args.batch_size
epochs = args.epochs
num_ensembles = args.num_ensembles

if gpus > 1:
    print('Using multi-GPU setup')
    strategy = tf.distribute.MirroredStrategy()
else:
    print('Using single-GPU setup')
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

with strategy.scope():
    #hyperparameter values
    learningrate=0.01
    momentum=0.1
    input_shape = (28, 28, 1)
    num_classes = 10

    #  to split the data of training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #https://www.educative.io/answers/how-to-merge-two-different-models-in-keras
    #https://stackoverflow.com/questions/61059918/could-not-compute-output-tensor-error-in-keras-functional-api

    # add a layer that returns the concatenation
    # of the positive part of the input and
    # the opposite of the negative part
    
    def create_model(input_shape, num_classes, input_name):
        input_layer = Input(shape=input_shape, name=input_name)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        return input_layer, output_layer

    def Std_Dev(v):
        a=1/3*((v[0]-v[3])^2+(v[1]-v[3])^2+(v[2]-v[3])^2)
        return a
    
    inputs = []
    outputs = []
    for i in range(num_ensembles):
        input_name = f'input_{i}'
        input_layer, output_layer = create_model(input_shape, num_classes, input_name)
        inputs.append(input_layer)
        outputs.append(output_layer)

    #Merging model A and B
    concatenated = concatenate(outputs)
    out = Dense(num_classes, activation='softmax', name='out_layer')(concatenated) #this one outputs the softmax
    out1 = Average()(outputs)
    #To add more outputs we need to modify the dataset and add two columns the ave and str_dv.
    #The average should be the same as the label value (SO NO NEED TO ADD TO DATASET) and the std dev Should be 0)
    #Maybe instead of adding this layer just calculate the std on the prediction so we provide the output of the concatenate layer
    #https://gist.github.com/akshaychawla/02849170e190fbd7fa9d431450e8d6ef ==Lambda with thee inputs
    #out2=Lambda(Std_Dev)([a_out,b_out,c_out,out1]) #<-needs to create this one with a lambda function

    #Model Definition
    new_combined = Model(inputs=inputs, outputs=[out,out1])

    new_combined.compile(loss=keras.losses.categorical_crossentropy,
          optimizer='adam',
          metrics=[['accuracy'], ['accuracy']])

    #train
    start_time = time.time()
    hist = new_combined.fit(
        {f'input_{i}': x_train for i in range(num_ensembles)},
        [y_train, y_train],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=({f'input_{i}': x_test for i in range(num_ensembles)}, [y_test, y_test])
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f'training time: {training_time} seconds')

    #evaluate
    start_time = time.time()
    score = new_combined.evaluate({f'input_{i}': x_test for i in range(num_ensembles)}, [y_test, y_test], verbose=1)
    end_time = time.time()
    eval_time = end_time - start_time
    print(f'eval time: {eval_time} seconds')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Test Accuracy Length: ', len(score))
    print(score)

