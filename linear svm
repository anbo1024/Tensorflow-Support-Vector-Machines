import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit



# data_set preparation
iris_data = datasets.load_iris()

x_vars = np.array([[x[1], x[3]] for x in iris_data['data']])
y_vars = np.array([[1.] if y == 0 else [-1.] for y in iris_data['target']])


sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=43)

for train_index, valid_index in sss.split(x_vars, y_vars.reshape(-1)):

    x_train, x_valid = x_vars[train_index], x_vars[valid_index]
    y_train, y_valid = y_vars[train_index], y_vars[valid_index]

#model setting
sess = tf.Session()
learning_rate = 0.001
batch_size = 60
epochs = 5000
hyper_c = 1000

# model structure
X_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal(shape=[2, 1], mean=0., stddev=0.2))
#b = tf.Variable(initial_value=[[0.]],)
b = tf.constant(1,dtype=tf.float32,shape=[1,1])
model_out = tf.add(tf.matmul(X_input, W), b)

l2_norm = 0.5*tf.reduce_sum(tf.square(W))
hinge_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(Y_target, model_out))))
hyper_parameter_C = tf.constant(hyper_c, shape=[1, 1], dtype=tf.float32)
loss = tf.add(l2_norm, tf.multiply(hyper_parameter_C, hinge_loss))

my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train_step = my_opt.minimize(loss)

# model metric

Y_pred = tf.sign(model_out)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_target, Y_pred),dtype=tf.float32))


# model initialization

init = tf.global_variables_initializer()
sess.run(init)

train_loss_list = []
train_acc_list = []

valid_loss_list = []
valid_acc_list = []

#model training

for epoch in range(epochs):

    # random shuffle
    index_list = list(range(len(x_train)))
    np.random.shuffle(index_list)
    x_train = x_train[index_list]
    y_train = y_train[index_list]

    for num in range(len(x_train)//batch_size):

        x_input = x_train[num*batch_size:batch_size*(num+1)]
        y_input = y_train[num*batch_size:batch_size*(num+1)]

        sess.run(train_step,feed_dict={X_input: x_input, Y_target: y_input})

        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X_input: x_input, Y_target: y_input})
        valid_loss, valid_acc = sess.run([loss, accuracy], feed_dict={X_input: x_valid, Y_target: y_valid})

        train_loss = np.squeeze(train_loss)
        valid_loss = np.squeeze(valid_loss)

        #print('training loss: {:.2f}, training acc: {:.2f}%'.format(train_loss, train_acc*100))
        #print('valid loss: {:.2f}, valid acc: {:.2f}%'.format(valid_loss, valid_acc*100))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

#visualization

train_num = len(train_loss_list)

#loss value visualization

plt.plot(range(train_num), train_loss_list, 'r-', label='train loss')
plt.plot(range(train_num), valid_loss_list, 'b--', label='valid loss')
plt.title('Loss Value')
plt.xlabel('train_step')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

#acc value visualization

plt.plot(range(train_num), train_acc_list, 'r-', label='train acc')
plt.plot(range(train_num), valid_acc_list, 'b--', label='valid acc')
plt.title('Accuracy Value')
plt.xlabel('train_step')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()

#data distribution

W_cal = sess.run(W)
b_cal = sess.run(b)
x_plot = np.linspace(2.,4.5,2)
y_plot = (-(x_plot*W_cal[0]+b_cal)/W_cal[1]).reshape(-1)
boundary_1 = (-(x_plot*W_cal[0]+b_cal+1)/W_cal[1]).reshape(-1)
boundary_2 = (-(x_plot*W_cal[0]+b_cal-1)/W_cal[1]).reshape(-1)


train_p = (y_train == 1).reshape(-1)
train_n = (y_train == -1).reshape(-1)
valid_p = (y_valid == 1).reshape(-1)
valid_n = (y_valid == -1).reshape(-1)

plt.plot(x_train[train_p, 0], x_train[train_p, 1], 'r*', label='Train pos')
plt.plot(x_train[train_n, 0], x_train[train_n, 1], 'b*', label='Train neg')
plt.plot(x_valid[valid_p, 0], x_valid[valid_p, 1], 'ro', label='valid pos')
plt.plot(x_valid[valid_n, 0], x_valid[valid_n, 1], 'bo', label='valid neg')
plt.plot(x_plot,y_plot,'g-')
plt.plot(x_plot,boundary_1,'y--')
plt.plot(x_plot,boundary_2,'y--')
plt.xlabel('feature #1')
plt.ylabel('feature #2')
plt.legend(loc='upper right')
plt.title('data distribution')
plt.show()
