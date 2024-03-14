import tensorflow as tf
from keras.datasets import imdb
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

def train():
    (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    x_train = vectorize_sequences(train_data)#进行ont-hot编码，转换成1*10000向量
    x_test = vectorize_sequences(test_data)#但是感觉这样的编码会丢失重复的值
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    history = model.fit(x_train,y_train,epochs=6,batch_size=512)
    results = model.evaluate(x_test,y_test)
    print(results)
    return results


def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequences in enumerate(sequences):
        results[i,sequences]=1.
    return results


def draw_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1,len(loss_values) + 1)

    plt.plot(epochs,loss_values,'bo',label='Training loss')
    plt.plot(epochs,val_loss_values,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()



if __name__ == "__main__":
    history = train()
    #draw_loss(history)