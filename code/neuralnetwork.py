import numpy as np
class Network:
    #intialization function
    def __init__(self,sizes):
         self.no_of_layers=len(sizes)
         self.sizes=sizes
         self.biases=[np.random.randn(y,1) for y in sizes[1:]]
         self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    #function to caluculate output of neural network
    def outputCaluculate(self,a):
        for w,b in zip(self.weights,self.biases):
            a=self.sigmoid(np.dot(w,a)+b)
        return a

    #function to caluculate sigmoid of a numpy array
    def sigmoid(self,input):
        return 1.0/(1.0+np.exp(-1*input))



    #function to caluculate sigmoid derivative of a numpy array
    def sigmoid_derivative(self,input):
        return self.sigmoid(input)*(1-self.sigmoid(input))


    #function to implement  gradient Descent
    def gradientDescent(self,no_of_iterations,training_data,test_data,learning_rate):
        for i in range(1,no_of_iterations):
            for data in training_data:
                self.update_weights_and_biases(data,learning_rate)
            print("For iteration no ",i," ",self.evaluate(test_data))


    #function to upgrade weights and biases according to gradient descent algorithm

    def update_weights_and_biases(self,data,learning_rate):
        gradient_weights=[np.zeros(np.shape(w))  for w in self.weights]
        gradient_biases=[np.zeros(np.shape(b))  for b in self.biases]
        x,y=data
        delta_w, delta_b = self.backPropagation(x,y)
        gradient_weights= [nb+dnb for nb, dnb in zip(gradient_weights, delta_w)]
        gradient_biases= [nw+dnw for nw, dnw in zip(gradient_biases, delta_b)]
        gradient_weights,gradient_biases=self.backPropagation(data[0],data[1])
        self.weights=[w-learning_rate*gw  for w,gw in zip(self.weights,gradient_weights)]
        self.biases=[b-learning_rate*gb  for b,gb  in zip(self.biases,gradient_biases) ]


    #function to caluculate error produced due to neural network
    def caluculateError(self,output_activation,y):
        return output_activation-y


    #function to caluculate gradients with respect to weights and biases
    def backPropagation(self,x,y):
        gradient_weights=[np.zeros(np.shape(w))  for w in self.weights]
        gradient_biases=[np.zeros(np.shape(b))  for b in self.biases]

        activation=x

        activations=[x]
        z=[]
        for w,b in zip(self.weights,self.biases):
            temp=np.dot(w,activation)+b
            z.append(temp)
            activation=self.sigmoid(temp)
            activations.append(activation)
        delta=self.caluculateError(activations[-1],y)*self.sigmoid_derivative(z[-1])
        gradient_weights[-1]=np.dot(delta,activations[-2].transpose())
        gradient_biases[-1]=delta

        for l in range(2,self.no_of_layers):
            temp=z[-l]
            sp=self.sigmoid_derivative(temp)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            gradient_biases[-l]=delta
            gradient_weights[-l]=np.dot(delta,activations[-l-1].transpose())
        return (gradient_weights,gradient_biases)



    #function to evaulate neural network by caluculating % of images correectly recognized
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.outputCaluculate(x)), y)
                        for (x, y) in test_data]
        return (sum(int(x == y) for (x, y) in test_results)/len(test_data))*100
