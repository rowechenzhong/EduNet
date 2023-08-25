# EduNet

In my freshman year at MIT, I wanted to take 6.036 machine learning, but didn't have enough credits. So I learned it myself. Then, to make sure I understood everything, I implemented all the major components of a neural network from scratch.

Later, I came back and supplemented the code with some commentary, features, and a few tutorials. If you're trying to use this as a resource, a good starting place might be the [Basic Dense Layer](Layers/Dense/Basic.md), the [Tutorial for its implementation](Layers/Dense/Basic-Implementation.md), and the accompanying [Source Code](Layers/Dense/Basic.py). The other parts are unpolished and not as well documented, go try google.


### Features

Neural Networks in EduNet are composed of many [Layers](Core/Layer.py) and a [Loss Function](Core/Loss.py) composed inside a [Model](Core/Model.py).

All Layers subclassing [Layer](Core/Layer.py) can be found in the [Layers](Layers) folder; these include various implementations of Dense, Convolution, and Recurrent layers.

The Adam optimizer was abstracted into the [AdamArray class](Util/AdamArray.py). This is similar to the paradigm found in PyTorch, where all of the back-propagation occurs under the hood, and users only need to worry about the forward propagation.

Various experiments can be found in the [Experiments](Experiments) folder. These include tests on MNIST, CIFAR-10, and some attempts at Deep Q-Learning. You can run any one of them as a script.

Our models are implemented using Numpy, which makes them rather slow. Some pre-trained models can be found in [Models](Models).