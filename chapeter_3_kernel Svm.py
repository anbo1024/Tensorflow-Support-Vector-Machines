import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# data_prepare
iris_data = datasets.load_iris()

y_data = iris_data['target'][iris_data['target'] != 2]
x_data = iris_data['data'][iris_data['target'] != 2, 1:3]

y_data = np.where(y_data == 0, 1, -1)

x_train, x_test, y_train, y_test = \
    train_test_split(
        x_data,
        y_data,
        test_size=0.33,
        random_state=42,
        shuffle=True)

y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1)


train_size = x_train.shape[0]
feature_dim = x_train.shape[1]
gamma_value = 0.5
epochs = 50

sess = tf.Session()
# stochastic batch training can not be used, the whole data should be trained together

# training part
X_train_input = tf.placeholder(shape=(train_size,feature_dim),dtype=tf.float32)
Y_target = tf.placeholder(shape=(train_size, 1), dtype=tf.float32)
alpha_weight = tf.Variable(np.random.rand(train_size,1) / 10,dtype=tf.float32)


# ensure the KKT
alpha_weight = tf.maximum(alpha_weight, 0)
gamma = tf.constant([gamma_value], dtype=tf.float32)

#weight_term calculate
weight_term = tf.reduce_sum(alpha_weight)

#kernel term calculate
x1 = tf.expand_dims(X_train_input, axis=1)
x2 = tf.expand_dims(X_train_input, axis=0)
x1 = tf.broadcast_to(x1, shape=(train_size, train_size, feature_dim))
x2 = tf.broadcast_to(x2, shape=(train_size, train_size, feature_dim))

    # linear kernel K<xi,xj> = xi dot xj
#kernel = tf.reduce_sum(tf.multiply(x1, x2), axis=2)

    # RBF kernel k<xi,xj> = exp(-gamma*(xi-xj)**2)
kernel = tf.exp(tf.multiply(-gamma, tf.reduce_sum((x1 - x2)**2, axis=2)))

    #
Y_alpha = tf.multiply(Y_target, alpha_weight)
Y_alpha_cross = tf.matmul(Y_alpha, tf.transpose(Y_alpha))

kernel_term = tf.reduce_sum(tf.multiply(Y_alpha_cross, kernel))

# loss function
loss = tf.subtract(tf.multiply(kernel_term, 0.5), weight_term)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_step = my_opt.minimize(loss)

# model initialization
init = tf.global_variables_initializer()
sess.run(init)

alpha_weight_list = []

for epoch in range(epochs):

    sess.run(train_step, feed_dict={X_train_input: x_train, Y_target: y_train})
    alpha_weight_cal = sess.run(alpha_weight)
    loss_cal = sess.run(loss,feed_dict={X_train_input: x_train,Y_target: y_train})

    # print(alpha_weight_cal)
    alpha_weight_list.append(alpha_weight_cal)

    print('When epoch %d' % epoch, loss_cal)

# plt.plot(alpha_weight_cal, '*b', label='alpha_weight')
# plt.title('Weight @')
# plt.show()

# get the support vector
support_select = (alpha_weight_cal > 0.05).reshape(-1)
x_support = x_train[support_select]
y_support = y_train[support_select]
support_weight = alpha_weight_cal[support_select]
support_size = x_support.shape[0]

# linear kernel
#support_kernel = np.matmul(support_x, np.transpose(support_x))

# RBF kernel k<xi,xj> = exp(-gamma*(xi-xj)**2)
support_x1 = np.expand_dims(x_support, 1)
support_x2 = np.expand_dims(x_support, 0)
support_x1 = np.broadcast_to(support_x1,shape=(support_size,support_size,feature_dim))
support_x2 = np.broadcast_to(support_x2,shape=(support_size,support_size,feature_dim))
support_kernel = np.exp((-gamma_value) * np.sum((support_x1 - support_x2)**2, axis=2))

# calculate the weight and bias
# linear kernel, the weight of the hyper plane

weight_cal = np.sum(alpha_weight_cal * y_train * x_train, axis=0)

# RBF kernel, the weight of every distance to support vector
weight_alpha_y_support = y_support * support_weight

support_kernel_sum = np.sum(support_kernel, axis=1).reshape(-1, 1)
b_cal = np.mean(1 / y_support - (support_weight *y_support * support_kernel_sum))

train_1 = (y_train == 1).reshape(-1)
train_0 = (y_train == -1).reshape(-1)
plt.plot(x_train[train_1, 0], x_train[train_1, 1], '*r')
plt.plot(x_train[train_0, 0], x_train[train_0, 1], '*b')
plt.plot(x_support[:, 0], x_support[:, 1], 'oy')
plt.show()


# Test part

# ops.reset_default_graph()
#sess = tf.Session()


def predict_model(x_support,y_support,weight_support,b_cal,x_test,y_test=None):

    test_size = x_test.shape[0]
    feature_dim = x_test.shape[1]
    support_size = x_support.shape[0]

    X_test_input = tf.placeholder(shape=(test_size, feature_dim), dtype=tf.float32)
    Y_test_input = tf.placeholder(shape=(test_size, 1), dtype=tf.float32)

    X_support_input = tf.placeholder(
        shape=(
            support_size,
            feature_dim),
        dtype=tf.float32)
    Y_support_input = tf.placeholder(shape=(support_size, 1), dtype=tf.float32)

    x3 = tf.expand_dims(X_test_input, axis=1)
    x4 = tf.expand_dims(X_support_input, axis=0)
    x3 = tf.broadcast_to(x3, shape=(test_size, support_size, feature_dim))
    x4 = tf.broadcast_to(x4, shape=(test_size, support_size, feature_dim))

    #linear kernel
    #test_kernel = tf.reduce_sum(tf.multiply(x3, x4), axis=2)

    #RBF kernel
    kernel_test = tf.exp(tf.multiply(-gamma, tf.reduce_sum((x3 - x4) ** 2, axis=2)))
    model_test = tf.matmul(kernel_test, tf.multiply(weight_support, y_support))
    bias_test = tf.constant(b_cal, shape=(test_size, 1), dtype=tf.float32)

    #
    ones = np.ones(shape=(test_size, 1), dtype=np.float32)
    minus_ones = -1 * ones
    Y_pred = tf.where(tf.add(model_test, bias_test) > 0, ones, minus_ones)


    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                Y_pred,
                Y_test_input),
            dtype=tf.float32))

    y_test_prd = sess.run(Y_pred,
                          feed_dict={
                              X_support_input: x_support,
                              Y_support_input: y_support,
                              X_test_input: x_test})


    if y_test is not None:

        test_acc = sess.run(accuracy, feed_dict={X_support_input: x_support,
                                             Y_support_input: y_support,
                                             X_test_input: x_test,
                                             Y_test_input: y_test})

        print('the prediction acc: {:.2f}%'.format(test_acc*100))

    return y_test_prd



y_test_prd = predict_model(x_support,y_support,support_weight,b_cal,x_test,y_test)
test_1 = (y_test_prd == 1).reshape(-1)
test_0 = (y_test_prd == -1).reshape(-1)




#contourf plot data created and predicted
xx1_min, xx1_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
xx2_min, xx2_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.arange(
        xx1_min, xx1_max, 0.02), np.arange(
            xx2_min, xx2_max, 0.02))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
grid_prd = predict_model(x_support,y_support,support_weight,b_cal,grid_points).reshape(xx1.shape)

print('grid shape')
print(grid_prd.shape)
plt.contourf(xx1, xx2, grid_prd, cmap=plt.cm.Paired)
plt.plot(x_train[train_1, 0], x_train[train_1, 1], '*r',label='train +1')
plt.plot(x_train[train_0, 0], x_train[train_0, 1], '*b',label='train -1')
plt.plot(x_test[test_1, 0], x_test[test_1, 1], '^r',label='test +1')
plt.plot(x_test[test_0, 0], x_test[test_0, 1], '^b',label='test -1')
plt.plot(x_support[:, 0], x_support[:, 1], 'oy',label='support data')
plt.legend(loc='upper right')
plt.show()
