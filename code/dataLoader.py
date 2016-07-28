from six.moves import cPickle
import gzip
import numpy as np
def load_data():
    fi = gzip.open('../mnsit_data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(fi,encoding='latin1')
    fi.close()
    return (training_data, validation_data, test_data)

def load_and_feature_data():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [make_array(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, test_data)

def make_array(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
