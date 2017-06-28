# this example comes from standford's cs224d lecture7, tensorflow tutorial
# https://cs224d.stanford.edu/lectures/
import numpy as np
# import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf


# define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)

# define data size and batch size
n_samples = 1000
batch_size = 100

# tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

# define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# define variables to be learned
with tf.variable_scope('linear-regression'):
    W = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pred)**2 / n_samples)

# sample code to run one step of gradient descent
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

best_para = {'W':0, 'b':0, 'loss_val':100}
with tf.Session() as sess:
    # initialize variables in graph
    sess.run(tf.initialize_all_variables())
    # gradient descent loop for 500 steps
    for i in range(5000):
        # select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        # do gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})
        if loss_val <= best_para['loss_val']:
            best_para['loss_val'] = loss_val
            best_para['W'] = W.eval()
            best_para['b'] = b.eval()
            print(W.eval(), b.eval())
            print(loss_val)

# plot input data
plt.scatter(X_data, y_data)
y_pred2 = X_data * best_para['W'] + best_para['b']
plt.plot(X_data, y_pred2, 'r--')
plt.show()










