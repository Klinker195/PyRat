import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# pyrat # pyrat - Rapid Artificial Training

import pyrat.datasets as rat_datasets
import pyrat.layers as rat_layers

if __name__ == "__main__":
    mnist_train_set = rat_datasets.MNIST(train=True)
    mnist_test_set = rat_datasets.MNIST(train=False)

    print(str(mnist_train_set))
    print()
    print(str(mnist_test_set))
    
    test = rat_layers.DenseLayer(input_size=10, output_size=10)
    
    