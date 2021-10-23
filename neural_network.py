import numpy as np
import math
import csv
import pickle
from random import random
from random import seed
from math import exp


# Funcion para inicializar los pesos en numeros aleatorios entre 0 y 1
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)], 'type': 'hidden'} for i in range(n_hidden)]
    network.append(hidden_layer)

    output_layer = [{'weights': [random() for i in range(n_hidden + 1)], 'type': 'output'} for i in range(n_outputs)]
    network.append(output_layer)

    return network

def activate(weights, inputs):
  activation = inputs[-1] #Se asume que el ultimo lugar es del bias
  for i in range(len(weights)-1):
    activation += weights[i] * inputs[i]
  return activation

def transfer(activation):
  return 1.0 / (1.0 + exp(-activation))

def relu(x):
    return max(0.0, x)

def forward_propagate(network, row):
  inputs = row

  for layer in network:
    new_inputs = []
    for neuron in layer:
      activation = activate(neuron['weights'], inputs)
      if neuron['type'] == 'output':
        neuron['output'] = transfer(activation)
      else:
        neuron['output'] = transfer(activation)
      new_inputs.append(neuron['output'])
    inputs = new_inputs
  return inputs

def relu_derivative(x):
  if x > 0:
    return 1
  else:
    return 0

def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i != len(network) - 1:  # Si no es la output layer
            for j in range(len(layer)):
                error = 0.0

                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])

                errors.append(error)

        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]

        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]

        for neuron in network[i]:

            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]

            neuron['weights'][-1] += learning_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
  for epoch in range(n_epoch):
    sum_error = 0
    for row in train:
      output = forward_propagate(network, row[0])
      output_decoded = 0 if output[0] < 0.5 else 1
      expected = row[1]
      sum_error += sum([(expected - output_decoded)**2])
      backward_propagate(network, expected)
      update_weights(network, row[0], l_rate)
    row = None
    print(f'>epoch={epoch}, lrate={l_rate}, error={sum_error}')

def predict(network, row):
  outputs = forward_propagate(network, row)
  return 0 if outputs[0] < 0.5 else 1

def datase_minmax(dataset):
    columns = []
    for sublist in dataset:
        for item in sublist:
            columns.append(float(item))
    stats = [min(columns), max(columns)]
    return stats

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[0]) / (minmax[1] - minmax[0])
    return dataset

if __name__ == "__main__":
    path = './'
    x = np.genfromtxt(path + 'X.csv', delimiter=',')
    y = np.genfromtxt(path + 'Y.csv', delimiter=',')
    minmax = datase_minmax(x)
    inputs = list()
    for row in x:
        inputs.append(list(row) + [1])  # Se le agrega el bias en la ultima posicion
    outputs = list()
    for row in y:
        outputs.append(int(row))  # Se formatean los outputs
    inputs = normalize_dataset(inputs, minmax)
    dataset = list(zip(inputs[:17000], outputs[:17000]))
    test_input = zip(inputs[17000:], outputs[17000:])
    test_output = [int(out) for out in y[-1000:]]
    test = list(zip(inputs[-2000:], outputs[-2000:]))  # zip(test_input, test_output)


    network = initialize_network(4, 1, 1)
    train_network(network, dataset, 0.25, 100, 1)


    correct_preds = 0
    for row in test:
        prediction = predict(network, row[0])
        print(f'Expected: {row[1]}, Got: {prediction}')
        if prediction == row[1]:
            correct_preds += 1

    print(f'Accuracy: {(correct_preds / len(test)) * 100}%')