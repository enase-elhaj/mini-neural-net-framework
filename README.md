Mini Neural Network Framework (NumPy)

This project implements a small, modular neural network framework using pure NumPy.
It is inspired by the design principles behind Keras and aims to deepen understanding of how neural networks work internally: layers, activations, loss functions, optimizers, forward pass, backpropagation, and training loops.

The framework is organized into reusable components that can be combined to build and train simple feedforward neural networks.
A demonstration notebook is included to show how the framework can be used to train a model on the digits dataset.

📌 Purpose of the Project

Provide a hands-on understanding of how neural networks operate under the hood

Build core components such as layers, activations, losses, and optimizers from scratch

Design a clean and modular API similar to Keras (Model, compile, fit, predict)

Strengthen both deep learning theory and software engineering skills

Offer a self-contained educational tool for experimenting with forward/backward propagation

📌 Features

Dense (fully connected) layer with trainable weights and biases

Activation layers including ReLU

Loss functions: Mean Squared Error, Categorical Cross Entropy, and a combined Softmax + Cross Entropy version

SGD optimizer with optional momentum

Model class supporting forward pass, backward pass, weight updates, and batch training

Support for mini-batch gradient descent and multiple epochs

Example notebook demonstrating training on the digits dataset

Reaches ~86% accuracy on the test set using a simple 2-layer neural network

📌 Project Structure
mini-neural-net-framework/
│── README.md
│── requirements.txt
│── src/
│   └── nn/
│       ├── __init__.py
│       ├── layers.py
│       ├── activations.py
│       ├── losses.py
│       ├── optimizers.py
│       └── model.py
│
└── notebooks/
    └── demo_mnist.ipynb

📌 Installation
git clone https://github.com/enase-elhaj/mini-neural-net-framework.git
cd mini-neural-net-framework
pip install -r requirements.txt

📌 Usage Example

Inside notebooks/demo_mnist.ipynb, the framework is used as follows:

from nn import Dense, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.optimizers import SGD
from nn.model import Model


Build a simple model:

model = Model([
    Dense(64, 32),
    ReLU(),
    Dense(32, 10)
])


Compile with loss and optimizer:

model.compile(
    loss=SoftmaxCrossEntropy(),
    optimizer=SGD(lr=0.05, momentum=0.9)
)


Train:

model.fit(X_train, y_train, epochs=40, batch_size=32)


Predict and evaluate:

predictions = model.predict(X_test)


The model achieves approximately 86% accuracy on the digits dataset.

📌 Dataset

The demo uses the digits dataset from scikit-learn:

8×8 images of handwritten digits

10 classes (0–9)

Included with scikit-learn (no download required)

📌 Future Enhancements

Additional activation functions (Sigmoid, Tanh, LeakyReLU)

Additional optimizers (Adam, RMSProp)

Regularization (Dropout, L2)

Model saving/loading

Unit tests for reliability

📌 Author

Developed by Enas Elhaj
Graduate Student, Applied AI & Data Science
University of Denver