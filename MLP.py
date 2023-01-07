import random
import warnings
import numpy as np
from preprocess import *
from helperfunctions import *
# suppress warnings
warnings.filterwarnings('ignore')


C1p = C12 = C13 = C2p = C21 = C23 = C3p = C31 = C32 = 0

baisAv = None
func = None
NumEpochs = None
learning_rate = None

def Setter(NEpochs, fun, LR, BaisValue):
    global baisAv, NumEpochs, learning_rate, func
    baisAv = BaisValue
    NumEpochs = NEpochs
    learning_rate = LR
    func = fun
 
def CreateHidden(Inputs, Layers, NeuronsPerLayer):
    AllHidden = []
    if baisAv: # check bais
        for i in range(Layers): # loop over each layer
            hiddenLayer = []
            for j in range(NeuronsPerLayer[i]):
                if i == 0: # First hidden layer
                    # create Neuron as Dict with weights as a list with random values with Number of Input adding one for bais
                    Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for x in range(Inputs+1)])
                else: # other Hidden layers
                # create Neuron as Dict with weights as a list with random values with Number of weights from number of neurons in previous layer adding one for bais
                    Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for x in range(NeuronsPerLayer[i - 1]+1)])
                hiddenLayer.append(Neuron)  # append neuron to hidden layer list
            AllHidden.append(hiddenLayer)  # append Hidden layer to hidden layers list
    else:
        for i in range(Layers): # loop over each layer
            hiddenLayer = []
            for j in range(NeuronsPerLayer[i]):
                if i == 0: # First hidden layer
                    # create Neuron as Dict with weights as a list with random values with Number of Input
                    Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for x in range(Inputs)])
                else: # other Hidden layers
                # create Neuron as Dict with weights as a list with random values with Number of weights from number of neurons in previous layer
                    Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for x in range(NeuronsPerLayer[i - 1])])
                hiddenLayer.append(Neuron)  # append neuron to hidden layer list
            AllHidden.append(hiddenLayer)  # append Hidden layer to hidden layers list
    return AllHidden

def CreateOutput(NumOutputs, NumWeights): # Create output layer
    OutputLayer = []
    if baisAv: # check bais
        for i in range(NumOutputs): # loop over each neuron in output layer
            # create Neuron as Dict with weights as a list with random values with Number of weights from number of neurons in previous layer adding one for bais
            Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for z in range(NumWeights + 1)]) 
            OutputLayer.append(Neuron) # add Neuron to layer
    else:
        for i in range(NumOutputs):
            # create Neuron as Dict with weights as a list with random values with Number of weights from number of neurons in previous layer
            Neuron = dict(weights=[round(random.uniform(-1, 1), 2) for z in range(NumWeights)])
            OutputLayer.append(Neuron) # add Neuron to layer
    return OutputLayer

def createNetwork(input, Layers, NeuronsPerLayer, Outputs): # Create the Whole Network
    hiddenLayers = CreateHidden(input, Layers, NeuronsPerLayer) # Create all Hidden layers
    network = hiddenLayers 
    outputLayer = CreateOutput(Outputs, NeuronsPerLayer[-1]) # Create output layer
    network.append(outputLayer) # Append output layer to Network
    return network

def Net(weights, inputs): # Calculate Net value for both weights and input of the layer
    if baisAv: # check bais
        net = weights[-1]  # Assume last weight is bias
    else:
        net = 0.0
    for i in range(len(inputs)):
        net += (weights[i] * inputs[i])
    return round(net, 5)  # weighted sum

def Activationfunction(net):
    if func == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-net))
    else:
        return np.tanh(net)

def FeedForward(network, data): # FeedForward Algorithm
    Layerinput = data
    for layer in network:
        newInputs = list()  # the value calculated from activation function to be used as Inputs for the next layer
        for neuron in layer:  # neuron['weights'] has the value of all weights of that neuron
            # put the weighted sum into activation function
            neuron['output'] = round(Activationfunction(Net(neuron['weights'], Layerinput)), 5)
            # and append output to the neuron list, so we can use later
            newInputs.append(neuron['output'])  # have a list to store the activation value for next use
        Layerinput = newInputs  # the value coming out of act func is the inputs for the next hidden layer
    output = MapOutput(Layerinput)  # [0.99, 0.2, 0.3] => [1, 0, 0]
    return output  # At the end it will return output

def ActivationfunctionDerivative(output):
    if func == "Sigmoid":
        return output * (1 - output)  # the derivative of sigmoid function
    else:
        return 1 - (output ** 2)  # the derivative of tanh function

def backpropagation(network, target): # Back Propagation Algorithm
    for i in reversed(range(len(network))):  # Loop over all the layers in the network Starting from the End
        layer = network[i]  # get a layer
        if i == len(network) - 1:  # output layer
            for j in range(len(layer)): # 0,0,1
                neuron = layer[j]
                neuron['localGradiant'] = round((target[j] - neuron['output']) * ActivationfunctionDerivative(neuron['output']), 5)
        else:  # hidden layers
            for j in range(len(layer)):  # j is the index of the neuron at that layer
                neuron = layer[j]
                Sum = 0.0
                for nextN in network[i + 1]:
                    # for the neuron in next layer, which has weights connected to current layer
                    Sum += (nextN['weights'][j] * nextN['localGradiant'])  # j -> index of the weights in next neuron
                neuron['localGradiant'] = round(Sum * ActivationfunctionDerivative(neuron['output']), 5)

def updateWeights(network, data): #  Weights updating Algorithm
    for i in range(len(network)): # loop over Network
        if i == 0: # if First hidden layer input is our features
            inputs = data # get the data
        else:  # for the hidden layer
            inputs = [n['output'] for n in network[i - 1]] # Get value of outputs from the previous layer
        for neuron in network[i]: # Loop over each neuron in current layer
            if baisAv: # Check if there is a bais to update it
                neuron['weights'][-1] += learning_rate * neuron['localGradiant']
                neuron['weights'][-1] = round(neuron['weights'][-1], 5)
            for j in range(len(inputs)):  # weight update for neurons
                neuron['weights'][j] += learning_rate * neuron['localGradiant'] * inputs[j]
                neuron['weights'][j] = round(neuron['weights'][j], 5)

def Fit(network, input, outputs):
    for e in range(NumEpochs): # loop over Epochs
        acc = 0
        for i in range(len(outputs)): # loop over each item in our training data
            row = MapRow(input.iloc[i]) # get the our input and map it to [f1, f2, f3, f4, f5]
            output = FeedForward(network, row) # Get output using FeedForward Algorithm
            target = MapTarget(outputs[i]) # Map target 
            backpropagation(network, target)
            updateWeights(network, row)
            if checkOutput(output, target): # calc accuracy if output == target
                acc += 1
    return acc

def Test(network, input, outputs): # predict the output for test data
    global C1p, C12, C13, C2p, C21, C23, C3p, C31, C32
    C1p = C12 = C13 = C2p = C21 = C23 = C3p = C31 = C32 = 0
    acc = 0
    for k in range(len(outputs)): # loop over each item in our testing data
        row = MapRow(input.iloc[k]) # get the our input and map it to [f1, f2, f3, f4, f5]
        output = FeedForward(network, row) # Get output using FeedForward Algorithm
        target = MapTarget(outputs[k]) # Map target 
        if checkOutput(output, target): # calc accuracy if output == target
            acc += 1
        if target ==  [1, 0, 0]:
            if checkOutput(output, target):
                C1p +=1 
            elif checkOutput(output,[0, 1, 0]):
                C12 += 1
            elif checkOutput(output,[0, 0, 1]):
                C13 += 1

        elif target ==  [0, 1, 0]:
            if checkOutput(output, target):
                C2p +=1 
            elif checkOutput(output,[1, 0, 0]):
                C21 += 1
            elif checkOutput(output,[0, 0, 1]):
                C23 += 1

        elif  target == [0, 0, 1]:
            if checkOutput(output, target):
                C3p +=1 
            elif checkOutput(output,[1, 0, 0]):
                C31 += 1
            elif checkOutput(output,[0, 1, 0]):
                C32 += 1

    return acc

def ConfusionMatrix(): # Function For adding Confusion Matrix data
    mat = createMatrix(4)
    mat[0][0]= "A|P"
    mat[1][0] = mat[0][1] = "Adelie"
    mat[2][0] = mat[0][2] = "Gentoo"
    mat[3][0] = mat[0][3] = "Chinstrap"
    mat[1][1] = C1p
    mat[2][2] = C2p
    mat[3][3] = C3p
    mat[1][2] = C12
    mat[1][3] = C13
    mat[2][1] = C21
    mat[2][3] = C23
    mat[3][1] = C31
    mat[3][2] = C32

    printMatrix(mat)
