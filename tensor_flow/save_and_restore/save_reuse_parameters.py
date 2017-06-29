# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"

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
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

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
    saver = tf.train.Saver()  # create a Saver node after all variable nodes are created
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch', epoch, 'MSE =', mse.eval())
                saver.save(sess=sess, save_path='cv/my_model.ckpt')
            sess.run(training_op)
        best_theta = theta.eval()
        print('The last MSE is', mse.eval())
        print('Best theta is', best_theta)
        saver.save(sess=sess, save_path='cv/my_model_final.ckpt')
# train_theta_by_gradient_descent(X_scaled, y)

def train_theta_by_gd_load_para(X, y):
    # global m
    # n_epochs = 1000  # 迭代次数
    # learning_rate = 0.01  # 之前学习率不能太大是因为X没有做缩放
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    # gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
    # training_op = tf.assign(theta, theta - learning_rate * gradients)
    # init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # create a Saver node after all variable nodes are created
    with tf.Session() as sess:
        saver.restore(sess, 'cv/my_model_final.ckpt')  # 加载上面保存的final check point
        # 在session中可以直接使用这些导入的参数进行运算
        mse, error, theta = sess.run([mse, error, theta])
        print(mse, theta)
        print('error is', error)
train_theta_by_gd_load_para(X_scaled, y)
