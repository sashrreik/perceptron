import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from random import random
from sklearn.metrics import confusion_matrix
from keras_visualizer import visualizer
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

def generate_dataset(num_samples, test_size = 0.3):

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # create a dataset with 2000 samples
    x_train, x_test, y_train, y_test = generate_dataset(2000, 0.3)

    # build model with 3 layers: 2 -> 5 -> 1
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(5, input_dim = 2, activation = "sigmoid"),
      tf.keras.layers.Dense(1, activation = "sigmoid")
    ])

    # choose optimiser
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)

    # compile model
    model.compile(optimizer = optimizer, loss = 'mse')

    # train model
    model.fit(x_train, y_train, epochs = 100)

    # evaluate model on test set
    print("\nEvaluation on the test set:")
    # model.evaluate(x_test,  y_test, verbose = 2)
    accuracy = model.evaluate(x_test, y_test)
    print("%0.3f" % accuracy)

    # y_pred = model.predict(x_test)
    # print(y_pred)
    # matrix = confusion_matrix(y_test,y_pred)
    # print('Confusion matrix : \n',matrix)

    # get predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    # print predictions
    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

    # save the model
    #model.save('saved_model')

    # load the model
    new_model = tf.keras.models.load_model('saved_model')

    # evaluate new model on test set
    print("\nEvaluation on the test set from new model:")
    # new_model.evaluate(x_test, y_test, verbose=2)
    accuracy = new_model.evaluate(x_test, y_test)
    print("%0.3f" % accuracy)

    # get predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = new_model.predict(data)

    # print predictions
    print("\nPredictions from new model:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

    # visualize the model
    visualizer(model, format='png', view=True)