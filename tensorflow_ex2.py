import tensorflow as tf

# 1.Build graph using TF operations
# H(x) = Wx + b
# X and Y data
#x_train = [1,2,3]
#y_train = [2,4,6]

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Our hypothesis Wx + b
hypothesis = x_train*W+b

# cost(W,b) = 1/m sigma(from 1~m) (H(x(i))-y(i))^2
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-y_train)) #reduce_mean = 1/m sigma(from 1~m)

# GradientDescent => minimize cost
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 2.Run/update graph and get results
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))

