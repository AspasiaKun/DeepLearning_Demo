import tensorflow as tf
import keras

class NWKernelRegression(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = (tf.Variable(initial_value=tf.random.uniform(shape=(1,))))
    
    def call(self, queries, keys, values, **kwargs):
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis=1)

        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
    