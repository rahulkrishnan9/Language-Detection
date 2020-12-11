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
        This class will store all the learned parameters and its call function will
        be what returns the logits for english vs german.
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


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: a batch of images of size 32x256
        :return: logits for german and english for every inputed image
        """
        #each image starts off as 32x256x1
        conv1 = tf.nn.conv2d(inputs, self.filter1, [1,1,1,1], padding = "SAME")
        conv1 = tf.nn.bias_add(conv1, self.bias1)
        moments1 = tf.nn.moments(conv1, [0,1,2])
        conv1 = tf.nn.batch_normalization(conv1, moments1[0],moments1[1],None,None,self.variance_epsilon)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding = "VALID")
        #now it's 16x128x32
        conv2 = tf.nn.conv2d(conv1, self.filter2, [1,1,1,1], padding = "SAME")
        conv2 = tf.nn.bias_add(conv2, self.bias2)
        moments2 = tf.nn.moments(conv2, [0,1,2])
        conv2 = tf.nn.batch_normalization(conv2, moments2[0],moments2[1],None,None,self.variance_epsilon)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding = "VALID")
        #now it's 8x64x64
        conv3 = tf.nn.conv2d(conv2, self.filter3, [1,1,1,1], padding = "SAME")
        conv3 = tf.nn.bias_add(conv3, self.bias3)
        moments3 = tf.nn.moments(conv3, [0,1,2])
        conv3 = tf.nn.batch_normalization(conv3, moments3[0],moments3[1],None,None,self.variance_epsilon)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, [1,2,1,1], [1,2,1,1], padding = "VALID")
        #now it's 4x64x128
        conv4 = tf.nn.conv2d(conv3, self.filter4, [1,1,1,1], padding = "SAME")
        conv4 = tf.nn.bias_add(conv4, self.bias4)
        moments4 = tf.nn.moments(conv4, [0,1,2])
        conv4 = tf.nn.batch_normalization(conv4, moments4[0],moments4[1],None,None,self.variance_epsilon)
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, [1,2,1,1], [1,2,1,1], padding = "VALID")
        #now it's 2x64x128
        conv5 = tf.nn.conv2d(conv4, self.filter5, [1,1,1,1], padding = "SAME")
        conv5 = tf.nn.bias_add(conv5, self.bias5)
        moments5 = tf.nn.moments(conv5, [0,1,2])
        conv5 = tf.nn.batch_normalization(conv5, moments5[0],moments5[1],None,None,self.variance_epsilon)
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.nn.max_pool(conv5, [1,2,1,1], [1,2,1,1], padding = "VALID")
        flattened_data = tf.squeeze(conv5)
        #Now we have a features list of length 64
        final = tf.nn.conv1d(flattened_data, self.windows, 1, padding = "VALID")
        #Here we've made guesses at every time step of window size 5 shifting 1 at a time, so the output is length 60
        final = tf.nn.bias_add(final, self.finalbias)
        final = tf.nn.max_pool(final, [1,4,1], [1,4,1], padding = "VALID")
        #After max pooling we have 15 logit sets for the image
        logits = tf.squeeze(tf.nn.avg_pool(final, [1,15,1], [1,1,1], padding = "VALID"))
        #We now take the average of them to get a guess of the image's language
        return logits


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)


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

def train(model, english_loader, german_loader, english_loader2, german_loader2):
    '''
    Trains the model on 100 batches
    takes in the four dataloaders
    '''
    for i in range(100):
        batch_english = english_loader.getNext()
        batch_german = german_loader.getNext()
        batch_english2 = english_loader2.getNext()
        batch_german2 = german_loader2.getNext()
        ones = tf.ones(50)
        zeros = tf.zeros(50)
        batch_labels = tf.cast(tf.concat([ones,zeros], 0),tf.int64)
        batch_inputs = tf.concat([batch_english, batch_english2, batch_german, batch_german2],0)
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

def test(model, english_loader, german_loader, english_loader2, german_loader2):
    '''
    Tests the model on 20 batches.
    Takes in the four data loaders
    returns the final accuracy
    '''
    accuracy = 0
    for i in range(20):
        batch_english = english_loader2.getNext()
        batch_german = german_loader2.getNext()
        batch_english2 = english_loader.getNext()
        batch_german2 = german_loader.getNext()
        ones = tf.ones(50)
        zeros = tf.zeros(50)
        batch_labels = tf.cast(tf.concat([ones,zeros], 0),tf.int64)
        batch_inputs = tf.concat([batch_english, batch_english2, batch_german, batch_german2],0)
        loader_shuffle = tf.random.shuffle(tf.range(100))
        batch_inputs = tf.gather(batch_inputs,loader_shuffle)
        batch_labels = tf.gather(batch_labels,loader_shuffle)
        predictions = model.call(tf.expand_dims(batch_inputs,axis=-1)) #calls the call function
        accuracy += model.accuracy(predictions, batch_labels)
        print (accuracy)
    return accuracy.numpy()/20.0

def main():
    english_loader = DataLoader('data/sentences/', 25, (32, 256))
    german_loader = DataLoader('data/GT4HistOCR/', 25, (32, 256))
    english_loader2 = DataLoader('data/english_generated/', 25, (32, 256))
    german_loader2 = DataLoader('data/something/', 25, (32, 256))
    model = Model()
    # plt.imshow(english_loader.getNext()[0])
    # plt.show()
    # plt.imshow(german_loader.getNext()[0])
    # plt.show()
    train(model, english_loader, german_loader, english_loader2, german_loader2)
    print(test(model,english_loader,german_loader, english_loader2, german_loader2))


if __name__ == '__main__':
	main()
