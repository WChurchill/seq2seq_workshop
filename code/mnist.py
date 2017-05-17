# import tensorflow
import tensorflow as tf
# import the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data

data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)

# the length of each input vector
# represents a flattened 28x28 greyscale image
INPUT_DIM = 784 
# length of output vector
# binary output for each number 0-9
OUTPUT_DIM = 10

with tf.Graph().as_default():
    # create a session that runs the graph operations
    sess = tf.Session()
    # tensorflow placeholders
    x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM],)

    # 'Variable' types are used for learned parameters
    # weight matrix variable
    W = tf.Variable(tf.zeros([INPUT_DIM, OUTPUT_DIM]),)
    # bias node variable
    bias1 = tf.Variable(tf.zeros([OUTPUT_DIM]))

    # define the procedure for computing a predicted value
    y = tf.matmul(x, W) + bias1

    # Loss function
    #
    # this defines how bad a single prediction is
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # Define learning rate
    learning_rate = tf.constant(0.1)
    max_iterations = 10000
    batch_size = 512

    # Define how parameters are adjusted
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(cross_entropy)

    # initializing all the parameters as zero is a bad idea.
    #
    # Use global_variables_initializer() to choose a smarter starting
    # point

    sess.run(tf.global_variables_initializer())
    # Train the model
    for i in range(max_iterations):
        batch = data_sets.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]}, session=sess)
        if i % 500 is 0:
            correct_prediction = tf.equal(tf.argmax(y, 1),
                                          tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Iteration ', i, end='\t:')
            print(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]},
                                session=sess))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('FINAL ACCURACY: ', end='')
    print(accuracy.eval(feed_dict={x: data_sets.test.images, y_:
                                   data_sets.test.labels},
                        session=sess))
