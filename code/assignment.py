from preprocess import preprocess, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpImg
import time
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.learning_rate = 0.001
        self.variance_epsilon = 0.00001 #same
        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,1,32], stddev=.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,32,64], stddev=.1))
        self.filter3= tf.Variable(tf.random.truncated_normal([3,3,64,128], stddev=.1))
        self.filter4 = tf.Variable(tf.random.truncated_normal([3,3,128,128], stddev=.1))
        self.filter5 = tf.Variable(tf.random.truncated_normal([3,3,128,256], stddev=.1))
        self.bias1 = tf.Variable(tf.random.truncated_normal([32], stddev=.1))
        self.bias2 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))
        self.bias3 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))
        self.bias4 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))
        self.bias5 = tf.Variable(tf.random.truncated_normal([256], stddev=.1))
        self.windows = tf.Variable(tf.random.truncated_normal([5,256,2], stddev=.1))
        self.finalbias = tf.Variable(tf.random.truncated_normal([2], stddev=.1))


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        conv1 = tf.nn.conv2d(inputs, self.filter1, [1,1,1,1], padding = "SAME")
        conv1 = tf.nn.bias_add(conv1, self.bias1)
        moments1 = tf.nn.moments(conv1, [0,1,2])
        conv1 = tf.nn.batch_normalization(conv1, moments1[0],moments1[1],None,None,self.variance_epsilon)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding = "VALID")
        conv2 = tf.nn.conv2d(conv1, self.filter2, [1,1,1,1], padding = "SAME")
        conv2 = tf.nn.bias_add(conv2, self.bias2)
        moments2 = tf.nn.moments(conv2, [0,1,2])
        conv2 = tf.nn.batch_normalization(conv2, moments2[0],moments2[1],None,None,self.variance_epsilon)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding = "VALID")
        conv3 = tf.nn.conv2d(conv2, self.filter3, [1,1,1,1], padding = "SAME")
        conv3 = tf.nn.bias_add(conv3, self.bias3)
        moments3 = tf.nn.moments(conv3, [0,1,2])
        conv3 = tf.nn.batch_normalization(conv3, moments3[0],moments3[1],None,None,self.variance_epsilon)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, [1,2,1,1], [1,2,1,1], padding = "VALID")
        conv4 = tf.nn.conv2d(conv3, self.filter4, [1,1,1,1], padding = "SAME")
        conv4 = tf.nn.bias_add(conv4, self.bias4)
        moments4 = tf.nn.moments(conv4, [0,1,2])
        conv4 = tf.nn.batch_normalization(conv4, moments4[0],moments4[1],None,None,self.variance_epsilon)
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, [1,2,1,1], [1,2,1,1], padding = "VALID")
        conv5 = tf.nn.conv2d(conv4, self.filter5, [1,1,1,1], padding = "SAME")
        conv5 = tf.nn.bias_add(conv5, self.bias5)
        moments5 = tf.nn.moments(conv5, [0,1,2])
        conv5 = tf.nn.batch_normalization(conv5, moments5[0],moments5[1],None,None,self.variance_epsilon)
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.nn.max_pool(conv5, [1,2,1,1], [1,2,1,1], padding = "VALID")
        flattened_data = tf.squeeze(conv5)
        final = tf.nn.conv1d(flattened_data, self.windows, 1, padding = "VALID")
        final = tf.nn.bias_add(final, self.finalbias)
        final = tf.nn.max_pool(final, [1,4,1], [1,4,1], padding = "VALID")
        logits = tf.squeeze(tf.nn.avg_pool(final, [1,15,1], [1,1,1], padding = "VALID"))
        return logits


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

def train(model, english_loader, german_loader):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    for i in range(100):
        batch_english = english_loader.getNext()
        batch_german = german_loader.getNext()
        ones = tf.ones(50)
        zeros = tf.zeros(50)
        batch_labels = tf.cast(tf.concat([ones,zeros], 0),tf.int64)
        batch_inputs = tf.concat([batch_english, batch_german],0)
        loader_shuffle = tf.random.shuffle(tf.range(100))
        batch_inputs = tf.gather(batch_inputs,loader_shuffle)
        batch_labels = tf.gather(batch_labels,loader_shuffle)
        with tf.GradientTape() as tape:
            predictions = model.call(tf.expand_dims(batch_inputs,axis=-1)) #calls the call function
            loss = model.loss(predictions, batch_labels)
            print(model.accuracy(predictions, batch_labels))
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, english_loader, german_loader):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    accuracy = 0
    for i in range(20):
        batch_english = english_loader.getNext()
        batch_german = german_loader.getNext()
        ones = tf.ones(50)
        zeros = tf.zeros(50)
        batch_labels = tf.cast(tf.concat([ones,zeros], 0),tf.int64)
        batch_inputs = tf.concat([batch_english, batch_german],0)
        loader_shuffle = tf.random.shuffle(tf.range(100))
        batch_inputs = tf.gather(batch_inputs,loader_shuffle)
        batch_labels = tf.gather(batch_labels,loader_shuffle)
        predictions = model.call(tf.expand_dims(batch_inputs,axis=-1)) #calls the call function
        accuracy += model.accuracy(predictions, batch_labels)
        print (accuracy)
    return accuracy.numpy()/20.0

def main():
    english_loader = DataLoader('data/sentences/', 50, (32, 256), 32)
    german_loader = DataLoader('data/GT4HistOCR/', 50, (32, 256), 32)
    batch = english_loader.getNext()
    model = Model()
    train(model, english_loader, german_loader)
    # plt.imshow(batch[0])
    # plt.show()
    print(test(model,english_loader,german_loader))


if __name__ == '__main__':
	main()
