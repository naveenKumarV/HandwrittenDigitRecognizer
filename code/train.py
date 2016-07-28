import dataLoader
import neuralnetwork
training_data,test_data=dataLoader.load_and_feature_data()
net=neuralnetwork.Network([784,30,10])
net.gradientDescent(30,training_data,test_data,0.5)
print(net.evaluate(test_data))
