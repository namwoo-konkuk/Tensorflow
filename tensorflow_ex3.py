import tensorflow as tf

# Now we can use X and Y in place of x_data and y_data
## placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Our hypothesis Wx + b
hypothesis = X*W+b

# cost(W,b) = 1/m sigma(from 1~m) (H(x(i))-y(i))^2
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y)) #reduce_mean = 1/m sigma(from 1~m)

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
    cost_val,W_val,b_val,_=\
        sess.run([cost,W,b,train],feed_dict={X:[1,2,3,4,5],Y:[2.1,3.1,4.1,5.1,6.1]})
    # same as cost_val,W_val,b_val,_= sess.run([cost,W,b,train],feed_dict={X:[1,2,3],Y:[1,2,3]})
    if step%20 == 0:
        print(step,cost_val,W_val,b_val)

# Testing our model
print(sess.run(hypothesis,feed_dict={X:[5]}))
print(sess.run(hypothesis,feed_dict={X:[2.5]}))
print(sess.run(hypothesis,feed_dict={X:[1.5,3.5]}))



