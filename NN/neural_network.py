import random
import numpy as np
import copy
import math
import timeit

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derv(x):
    return np.multiply(x ,( 1.0 - x))

class neuron:
    def __init__(self,prev_neurons, weights = None, bias = None):
        self.inputs = []
        self.input = 0
        self.output = 0
        self.error = 0
        self.prev_neurons = prev_neurons
        self.weights = []

        if weights == None:
            for _ in range(self.prev_neurons):
                self.weights.append(random.uniform(0,1))
        else:
            self.weights = weights
        
        if bias == None:
            self.bias = random.uniform(0,1)
        else:
            self.bias = bias 

    def feedforward(self,input_values):
        self.inputs = input_values
        self.input = np.dot(input_values,self.weights)
        self.input = self.input + self.bias
        self.output = sigmoid(self.input)
        return self.output
    
    def update(self,learn_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learn_rate * self.error * self.inputs[i]
        self.bias += learn_rate * self.error
        

class  neural_network:
    def __init__(self,input_layer_size, output_layer_size, learn_rate=0.1):
        self.output_layer = []
        self.hidden_layer = []
        self.predicted_outputs = []
        self.learn_rate = learn_rate
        for i in range(input_layer_size):
            self.hidden_layer.append(neuron(input_layer_size))
        for i in range(output_layer_size):
            self.output_layer.append(neuron(len(self.hidden_layer)))
        

    def run_once(self,input_data):
        temp = []
        prev_layer_outputs =[]
        
        for neuron in self.hidden_layer:
            temp.append(neuron.feedforward(input_data))
        
        prev_layer_outputs = copy.deepcopy(temp)
        temp.clear()

        for neuron in self.output_layer:
            temp.append(neuron.feedforward(prev_layer_outputs))

        prev_layer_outputs = copy.deepcopy(temp)
        return prev_layer_outputs

    def backpropagation(self,actual_outputs):
        #calculate error in output layer
        for i in range(len(self.output_layer)):
            self.output_layer[i].error = (actual_outputs[i] - self.output_layer[i].output) * sigmoid_derv(self.output_layer[i].output)
        #calculate error in hidden layer dit nog fixen
        for n in range(len(self.hidden_layer)):
            self.hidden_layer[n].error = 0.0
            for i in range(len(self.output_layer)):
                self.hidden_layer[n].error += (self.output_layer[i].weights[n] * self.output_layer[i].error) * sigmoid_derv(self.hidden_layer[n].output)

    def update_neurons(self):
        for neuron in self.output_layer:
            neuron.update(self.learn_rate)
        for neuron in self.hidden_layer:
            neuron.update(self.learn_rate)

    def train(self,input_data_set,expected_outputs,itterations=10000):
        for itter in range(itterations):
            for item in range(len(input_data_set)):
                self.run_once(input_data_set[item])
                self.backpropagation(expected_outputs[item])
                self.update_neurons()



def load_data(file_name):
    return np.genfromtxt(file_name, delimiter=',', usecols=[0,1,2,3])

def load_labels(file_name):
    return np.genfromtxt(file_name, delimiter=',', usecols=[4],dtype=str)

def labels_to_NN_output(labels):
    outputs =[]
    for label in labels:
        output =[]
        if label == "Iris-setosa":
            output = [1,0,0]
        elif label == "Iris-versicolor":
            output = [0,1,0]
        elif label == "Iris-virginica":
            output = [0,0,1]
        outputs.append(output)
    return outputs

def split_data_set(outputs,data):
    train_data =[]
    train_outputs =[]
    test_data =[]
    test_outputs =[]
    for i in range(len(data)):
        if i %2 == 0:
            train_data.append(data[i])
            train_outputs.append(outputs[i])
        else:
            test_data.append(data[i])
            test_outputs.append(outputs[i])
    return [[train_data,train_outputs],[test_data,test_outputs]]

def compare(predicted,actual):
    one_index = predicted.index(max(predicted))
    for i in range(len(predicted)):
        if i == one_index:
            predicted[i]=1
        else:
            predicted[i]=0
    return predicted == actual


def test_iris_NN(NN, test_data, test_outpus):
    total = len(test_data)
    good = 0
    fault = 0
    for i in range(total):
        predicted = NN.run_once(test_data[i])
        if compare(predicted,test_outpus[i]):
            good += 1
        else:
            fault += 1
    # return (aantal in test_set,aantal goed, aantal fout, percentage goed)
    return [total,good,fault,round(good/total * 100,1)]

def run_iris():
    print("loading iris data")
    data = load_data("iris.data")
    labels = load_labels("iris.data")
    outputs = labels_to_NN_output(labels)
    print("splitting data set in 2 parts")
    split_sets = split_data_set(outputs,data)

    train_data = split_sets[0][0]
    train_outpus = split_sets[0][1]

    test_data = split_sets[1][0]
    test_outpus = split_sets[1][1]

    number_of_itterations = 10000
    iris_NN = neural_network(4,3)
    #the network consists of 4 input neurons, 1 hidden layer containing 4 neurons and a outputlayer containing 3 neurons
    #since there are 3 flower types the output should be 3 neurons
    #since each data point had 4 parameters the input should be 4 neurons

    print("train network with ", number_of_itterations, " using the train set")
    iris_NN.train(train_data,train_outpus,number_of_itterations)
    print("testing network with test data set")
    test_results = test_iris_NN(iris_NN,test_data,test_outpus)

    print("test results:")
    print("entry's         : ",test_results[0])
    print("good predictions: ",test_results[1])
    print("bad  predictions: ",test_results[2])
    print("good  Percentage: ",test_results[3],"%")

def run_XOR():
    print("XOR validation:")
    #having a higher learn rate make the set learn quicker
    test = neural_network(2,1,0.3)
    test_set = ([1,1],[1,0],[0,0],[0,1])
    output = ([0],[1],[0],[1])
    # the XOR needs a lot more itterations or a higher learn rate before it has learned correctly
    # this is a known problem with a sigmoid based small network that needs to learn this problem
    # sometimes even 100.000 itterations is not enough
    test.train(test_set,output,100000)
    print("input: 0,0  expected = 0  predicted: ",round(test.run_once([0,0])[0],2))
    print("input: 1,0  expected = 1  predicted: ",round(test.run_once([1,0])[0],2))
    print("input: 0,1  expected = 1  predicted: ",round(test.run_once([0,1])[0],2))
    print("input: 1,1  expected = 0  predicted: ",round(test.run_once([1,1])[0],2))

def main():
    start = timeit.default_timer()
    run_iris()
    end = timeit.default_timer()
    print("time needed     : ",round(end-start,1), " secconds")
    
    print()

    start = timeit.default_timer()
    run_XOR()
    end = timeit.default_timer()
    print("XOR time needed: ",round(end-start,1), " secconds")

if __name__ == "__main__":
    main()