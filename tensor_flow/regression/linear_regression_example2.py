# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"
"""
对一个学习算法来说，最重要的有四个要素：
- 数据（训练数据和测试数据）
- 模型（用于预测或分类）
- 代价函数（评价当前参数的效果，对其求导可以计算梯度）
- 优化器（优化代价函数的参数，执行梯度下降）
Beter, 20170628
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  #  用于数据缩放


housing = fetch_california_housing()
m, n = housing.data.shape  # m是样本数，n是特征的数量
print(m, n)
# Gradient Descent requires scaling the feature vectors first
# X的缩放对后面的训练过程影响非常大，经过缩放的数据经过很少的迭代次数就可以收敛，学习率可以设得很大
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X_scaled = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X_scaled')

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# 方法1：使用正规方程直接求参数
def get_theta_by_normal_equation(X, y):
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)

# 方法2：梯度下降法训练参数（手动求导）
def train_theta_by_gradient_descent(X, y):
    global m
    n_epochs = 1000  # 迭代次数
    learning_rate = 0.01  # 之前学习率不能太大是因为X没有做缩放
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('Epoch', epoch, 'MSE =', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print('Best theta is', best_theta)
# train_theta_by_gradient_descent(X_scaled, y)

# 方法3：梯度下降法训练参数（自动求导）
def train_theta_by_autodiff(X, y):
    global m
    n_epochs = 10000
    learning_rate = 0.0000003  # 学习率不能太大
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    # gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
    gradients = tf.gradients(ys=mse, xs=[theta])[0]  # 自动对代价函数求导，代价函数是参数theta的函数
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('Epoch', epoch, 'MSE =', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print('Best theta is', best_theta)
# train_theta_by_autodiff(X, y)

# 方法4：使用优化器训练参数（Optimizer）
def train_theta_by_optimizer(X, y):
    global m
    n_epochs = 10000
    learning_rate = 0.0000003  # 学习率不能太大
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

    # gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
    # gradients = tf.gradients(ys=mse, xs=[theta])[0]  # 自动对代价函数求导，代价函数是参数theta的函数
    # training_op = tf.assign(theta, theta - learning_rate * gradients)

    # using an optimizer
    # 效果跟上面的方法差不多，只是更加简便
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # gradient descent optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9) # momentum optimizer
    training_op = optimizer.minimize(mse)  # 用优化器最小化代价函数mse

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('Epoch', epoch, 'MSE =', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print('Best theta is', best_theta)
# train_theta_by_optimizer(X, y)


# 方法5：使用小批次梯度下降(Mini-batch Gradient Descent)方法训练参数
# 实际上是随机梯度下降法，因为每次只使用一部分数据，因此训练过程中cost不是逐渐减小，而是会出现一定的波动
# 这里有两个很重要的概念：epoch and batch
# - batch: 由于每次训练只取全部样本中的一部分，因此一个batch只是一小部分训练样本(batch size定义了一个batch中样本的个数)
# - batch number: batch的个数是由样本数和batch size决定的，int(np.ceil(m / batch_size))
# - epoch: 是指使用全部训练样本训练一次，理论上按照batch size的大小，取int(np.ceil(m / batch_size))可以取完所有的样本
# - epoch number: 是指使用全部样本训练几次
# 使用placeholder nodes传递数据
def train_theta_by_mini_batch_gd():
    global m, scaled_housing_data_plus_bias, housing
    n_epochs = 100
    learning_rate = 0.00001
    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

    # using an optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9) # momentum optimizer
    training_op = optimizer.minimize(mse)  # 用优化器最小化代价函数mse

    init = tf.global_variables_initializer()

    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)  # 每次生成不同的seed
        indices = np.random.randint(m, size=batch_size)  # 随机生成batch_size个小于m的整数
        X_batch = scaled_housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            loss = 0
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                # 在这里传入mse, 可以返回mse的值
                _, loss = sess.run([training_op, mse], feed_dict={X: X_batch, y: y_batch})
            if epoch % 10 == 0:
                print('Epoch', epoch, 'MSE =', loss)
        best_theta = theta.eval()
        print('Best theta is', best_theta)
train_theta_by_mini_batch_gd()