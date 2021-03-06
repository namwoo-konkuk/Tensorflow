import tensorflow as tf

# Computational Graph
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print("node1:",node1, "node2:",node2)
print("node3: ",node3)

sess = tf.Session()
print("sess.run(node1,node2): ",sess.run([node1,node2]))
print("sess.run(node3): ",sess.run(node3))

# Placeholer
a = tf.placeholder(tf.float32) # make empty value
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node,feed_dict={a:3,b:4.5})) # feed_dict: feed value
print(sess.run(adder_node,feed_dict={a:[1,3],b:[2,4]}))

