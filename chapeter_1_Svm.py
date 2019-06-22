import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()

x_vals = np.linspace(0,10,100)
y_vals = 2*x_vals + 3 + np.random.normal(0,3,100)

x_vals = x_vals.reshape(-1,1)
y_vals = y_vals.reshape(-1,1)


learning_rate = 0.005
batch_size = 12
epochs = 50

x_data = tf.placeholder(dtype=tf.float32,shape=[None,1])
y_target = tf.placeholder(dtype=tf.float32,shape=[None,1])

A = tf.Variable(initial_value=np.random.normal(0,1,size=[1,1]),dtype=tf.float32)
b = tf.Variable(initial_value=np.random.normal(0,1,size=[1,1]),dtype=tf.float32)

y_pre = tf.add(tf.matmul(x_data,A),b)
L2_constrain = tf.reduce_mean(tf.square(A))
loss = tf.reduce_mean(tf.square(y_pre-y_target))

demming_denominator = tf.sqrt(tf.add(tf.square(A),1))
dim_loss = tf.reduce_mean(tf.truediv(tf.abs(y_pre-y_target),demming_denominator))

init = tf.global_variables_initializer()

sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = my_opt.minimize(loss)

L2_loss_list = []
dim_loss_list = []
for i in range(epochs):
    rand_index = np.random.choice(len(x_vals),size=batch_size)
    x_train = x_vals[rand_index]
    y_train = y_vals[rand_index]

    sess.run(train_step,feed_dict={x_data:x_train,y_target:y_train})
    train_loss = sess.run(loss,feed_dict={x_data:x_train,y_target:y_train})

    print('the training L2 loss:{:.2f}'.format(train_loss))
    L2_loss_list.append(train_loss)

A_l2 = sess.run(A)
b_l2 = sess.run(b)
y_vals_pre_l2 = A_l2*x_vals + b_l2

print(A_l2,b_l2)

sess.run(init)

for i in range(epochs):
    rand_index = np.random.choice(len(x_vals),size=batch_size)
    x_train = x_vals[rand_index]
    y_train = y_vals[rand_index]

    sess.run(train_step,feed_dict={x_data:x_train,y_target:y_train})
    train_loss = sess.run(dim_loss,feed_dict={x_data:x_train,y_target:y_train})

    print('the training dim loss:{:.2f}'.format(train_loss))
    dim_loss_list.append(train_loss)

A_dim = sess.run(A)
b_dim = sess.run(b)
y_vals_pre_dim = A_dim*x_vals + b_dim

print(A_dim,b_dim)


plt.plot(range(epochs),L2_loss_list,'r--',label='L2 loss')
plt.plot(range(epochs),dim_loss_list,'r-',label='dim loss')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Different Loss')
plt.show()


plt.plot(x_vals,y_vals,'r*',label='raw data')
plt.plot(x_vals,y_vals_pre_l2,'b--',label='l2 loss')
plt.plot(x_vals,y_vals_pre_dim,'g-*',label='dim loss')
plt.xlabel('x_var')
plt.ylabel('y_var')
plt.title('Regression')
plt.legend(loc='upper right')
plt.show()

