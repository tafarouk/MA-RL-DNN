#the following code is taken from Tariq Rashid's site for studying purpose

import numpy
#scipy.spcial for the sigmoid function expit()
import scipy.special


#neural network class definition
class neuralNetwork:

    #initialize the neural network
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden and output layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #learning rate
        self.lr=learningrate

        #initialize weights
        #pow riases a number to the power of second one
        #numpy.random.normal generates random numbers along spcific mean and standard deviation
        #in the shap of matrix spicified dimensions
        self.wih=numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        #set the activation function to sigmoid function
        self.activation_function=lambda x: scipy.special.expit(x)
        pass


    #train the neural network
    def train(self, inputs_list,targets_list):

        #convert inputs list to 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        #calculate the signals entering the final output layer
        final_inputs=numpy.do(self.who, hidden_outputs)

        #calculate the output signals
        final_outputs=self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors=targets-final_outputs

        # hidden layer error is the output_errors, split by weights, recombined
        # at hidden nodes
        hidden_errors=numpy.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs))
                                    ,numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs))
                                   ,numpy.transpose(inputs))


        pass


    #query the nerual network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T

        #calculate singals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)

        #caculate the singals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)

        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

        pass
