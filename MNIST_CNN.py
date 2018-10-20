from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)


import tensorflow as tf


def create_placeholders(num_input, num_classes):
    
    X = tf.placeholder(tf.float32, [None, num_input], name="X") #None is too not define the number of training examples
    Y = tf.placeholder(tf.float32, [None, num_classes], name="Y")

    return X, Y

def initialize_parameters():

    #No need to initialize biases because they are taken care of by tensorflow functions

    W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [7*7*64, 1024], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1024])
    W4 = tf.get_variable("W4", [1024, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [1, num_classes])

    # Add summary ops to collect data
    W1_h = tf.summary.histogram("weights 1", W1)
    W2_h = tf.summary.histogram("weights 2", W2)
    W3_h = tf.summary.histogram("weights 3", W3)
    b3_h = tf.summary.histogram("bias 3", b3)
    W4_h = tf.summary.histogram("weights 4", W4)
    b4_h = tf.summary.histogram("bias 4", b4)

    parameters = {
        "W1" : W1,
        "W2" : W2,
        "W3" : W3,
        "b3" : b3,
        "W4" : W4,
        "b4" : b4,
    }

    return parameters

def forward_propagation(X, parameters):
    # CONV2D --> RELU --> MAXPOOL --> CONV2D --> RELU --> MAXPOOL --> FLATTEN --> FULLYCONNECTED

    #Reshaping X
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    # Convolution Layer
    Z1 = tf.nn.conv2d(X,W1, strides=[1,1,1,1], padding='SAME')
    # ReLU
    A1 = tf.nn.relu(Z1)
    # Max Pooling Layer
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    # Convolution Layer
    Z2 = tf.nn.conv2d(P1,W2, strides=[1,1,1,1], padding='SAME')
    # ReLU
    A2 = tf.nn.relu(Z2)
    # Max Pooling Layer
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    # Reshape the output of the convolution and Max Pooling steps to fit fully connected layer input
    P2 = tf.contrib.layers.flatten(P2)  #Alternative: P2 = tf.reshape(P2, [-1, W3.get_shape().as_list()[0]])

    Z3 = tf.add(tf.matmul(P2, W3), b3)                                            
    A3 = tf.nn.relu(Z3) 
    Z4 = tf.add(tf.matmul(A3, W4), b4)                                           

    # alternative for fully connected layer
    #Z3 = tf.contrib.layers.fully_connected(P2, num_classes, activation_fn=None)

    return Z4

def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

    tf.summary.scalar("cost_function", cost)

    return cost


def model(learning_rate, num_epochs, batch_size, display_step):

    # Create Placeholders of shape (num_input, num_classes)
    X, Y = create_placeholders(num_input, num_classes)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation
    Z3 = forward_propagation(X, parameters)

    # Cost function(using cross entropy loss function)
    cost = compute_cost(Z3, Y)

    # Backpropagation(using AdamOptimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Compute the correct predictions
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(Z3), 1), tf.argmax(Y, 1))

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(init)

        summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

        for epoch in range(num_epochs):

            num_minibatches = int(mnist.train.num_examples/batch_size)
            epoch_cost = 0.
            for i in range(num_minibatches):
                #Retrieve the next mini_batch
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization operation + compute cost
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                epoch_cost += minibatch_cost / num_minibatches

                if i % 10 == 0:
                    print ("Cost after step %i: %f" % (i, epoch_cost))
                    print("Testing Accuracy:", \
                    sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256]}))
            
                # Write logs for each iteration
                summary_str = sess.run(merged_summary_op, feed_dict={X: batch_x, Y: batch_y})
                summary_writer.add_summary(summary_str, epoch*num_minibatches + i)

            if epoch % display_step == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("Training Accuracy:", \
                sess.run(accuracy, feed_dict={X: mnist.train.images[:256],
                                            Y: mnist.train.labels[:256]}))

        print("training Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.train.images[:256],     #if you try with all the dataset your memory will be too full
                                      Y: mnist.train.labels[:256]}))
        print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]}))

            

#hyper-parameters
learning_rate = 0.0003
num_epochs = 1 
batch_size = 128

display_step = 1 #in epochs

# Network parameters
num_input = 784
num_classes = 10

parameters = model(learning_rate, num_epochs, batch_size, display_step)

"""

Results: Testing Accuracy  : 0.996
         Training Accuracy : 0.997
        
"""