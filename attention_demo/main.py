import tensorflow as tf
from d2l import tensorflow as d2l
import numpy
import keras
from nw_kernel_regression import NWKernelRegression

def show_heatmaps(matrices, xlabel, ylabel, titles=None, 
                  figsize=(2.5, 2.5), cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def random_gen():
    n_train = 50
    x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))

    def f(x):
        return 2*tf.sin(x) + x**0.8
    
    y_train = f(x_train) + tf.random.normal((n_train,), 0.0, 0.5)
    x_test = tf.range(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)

    def plot_kernel_reg(y_hat):
        d2l.plot(x_test, [y_truth, y_hat],'x','y', lengend=['Truth','Pred'],
                 xlim=[0,5],ylim=[-1,5])
        d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

    X = tf.ones((2,1,4))
    Y = tf.ones((2,4,6))
    tf.matmul(X,Y).shape

    weights = tf.ones((2,10))*0.1
    values = tf.reshape(tf.range(20.0), shape = (2, 10))
    tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()

    X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
    Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
    keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
    values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
    
    net = NWKernelRegression()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        with tf.GradientTape() as t:
            loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
        grads = t.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
        animator.add(epoch + 1, float(loss))

    kays = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
    values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
    y_hat = net(x_test, keys, values)
    plot_kernel_reg(y_hat)

def main():
    # attention_weights = tf.reshape(tf.eye(10), (1,1,10,10))
    # show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    random_gen()


if __name__ == "__main__":
    main()