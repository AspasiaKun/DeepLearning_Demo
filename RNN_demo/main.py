import matplotlib.pyplot as plt
import math
import collections
import random
import re
import tensorflow as tf
import keras
from d2l import tensorflow as d2l
from vocab import Vocab
from seq_data_loader import SeqDataLoader
from rnn_model_scratch import RNNModelScratch
import numpy

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL+'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

def sequence_gen():
    T = 1000
    time = tf.range(1, T + 1, dtype=tf.float32)
    x = tf.sin(0.01 * time) + tf.random.normal([T], 0, 0.2)
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

def load_data_corpus_time_machine(batch_size, num_steps,
                                    user_random_iter=False, max_tokens=10000):
    
    data_iter = SeqDataLoader(
        batch_size, num_steps, user_random_iter, max_tokens)
    return data_iter, data_iter.vocab
    
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)
    #输入层
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens,num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    #输出层
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params

def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1, W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) +b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)

    return tf.concat(outputs, axis=0), (H,)

def predict_ch8(prefix, num_preds, net, vocab):
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: tf.reshape(tf.constant([outputs[-1]]),
                                   (1,1)).numpy()
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(grads, theta):
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad**2)).numpy()
                            for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad

def train_epoch_ch8(net, train_iter, loss, updater, user_random_iter):
    state, timer = None, d2l.Timer()
    metrix = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or user_random_iter:
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = tf.reshape(tf.transpose(Y), (-1))
            l = loss(y,y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))

        metrix.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metrix[0] / metrix[1]), metrix[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy, user_random_iter=False):
    with strategy.scope():
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        updater = keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, user_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

def main():
    batch_size, num_steps = 32,35
    train_iter, vocab = load_data_corpus_time_machine(batch_size, num_steps)
    train_random_iter, vocab_random_iter = load_data_corpus_time_machine(batch_size, num_steps, True)

    tf.one_hot(tf.constant([0,2]), len(vocab))
    X = tf.reshape(tf.range(10),(2,5))
    tf.one_hot(tf.transpose(X),28).shape

    device_name = d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)

    num_hiddens = 512
    num_epochs, lr = 500, 1.
    with strategy.scope():
        net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn, get_params)
    
    train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy, user_random_iter=True)
    state = net.begin_state(X.shape[0])
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)

if __name__ == "__main__":
    main()