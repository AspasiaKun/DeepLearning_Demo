import matplotlib.pyplot as plt
import math
import collections
import random
import re
import tensorflow as tf
import keras
from d2l import tensorflow as d2l

def import_data():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def sequence_gen():
    T = 1000
    time = tf.range(1, T + 1, dtype=tf.float32)
    x = tf.sin(0.01 * time) + tf.random.normal([T], 0, 0.2)
    #d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    plt.plot(time,x)

    tau = 4
    features = tf.Variable(tf.zeros((T - tau, tau)))
    for i in range(tau):
        features[:, i].assign(x[i: T - tau + i])
    labels = tf.reshape(x[tau:],(-1,1))

    batch_size, n_train = 16, 600
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),batch_size, is_train=True)
    
    net = get_net()
    loss = keras.losses.MeanSquaredError()
    train(net,train_iter,loss,5,0.01)

    onestep_preds = net(features)

    multistep_preds = tf.Variable(tf.zeros(T))
    multistep_preds[:n_train + tau].assign(x[:n_train + tau])
    for i in range(n_train + tau, T):
        multistep_preds[i].assign(tf.reshape(net(tf.reshape(multistep_preds[i - tau: i],(1, -1))),()))

    d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.numpy(), onestep_preds.numpy(),
          multistep_preds[n_train + tau:].numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
    #plt.plot(time,x,onestep_preds,multistep_preds)
    
    

def get_net():
    net = keras.Sequential([keras.layers.Dense(10, activation='relu'),
                            keras.layers.Dense(1)])
    return net

def train(net, train_iter, loss, epochs, lr):
    trainer = keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        print(f'epoch {epoch + 1},'f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

if __name__ == "__main__":
    sequence_gen()