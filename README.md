# Handwritten_Digit_Recognition_CNN
A ConvNet for handwritten digit recognition.

## Network Architecture

</br>
<p align="center">
  <img src="https://res.mdpi.com/entropy/entropy-19-00242/article_deploy/html/images/entropy-19-00242-g001.png" height=300px>
</p>
</br>
<p align="center" >
  <b>CONV2D</b> --> <b>RELU</b> --> <b>MAXPOOL</b> --> <b>CONV2D</b> --> <b>RELU</b> --> <b>MAXPOOL</b> --> <b>FLATTEN</b> --> <b>FULLYCONNECTED</b>
</p>
</br>

#### Hyperparameters used

- learning rate: 0.001
- epochs: 1
- batch size: 128

## Dataset

The network is trained on [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits. The dataset contains 60'000 examples for training and 10'000 examples for testing. the handwritten digits come as 28x28 fixed-size images with values ranging from 0 to 1, only one Channel.

## Results

- Training Accuracy: 0.997
- Testing Accuracy: 0.996

## Details

- It uses tensorflow.
- To train: ```python MNIST_CNN.py```
- A tensorflow summary is included with the evolution of the weights, biases ,and costs.
- To access tensorboard: ```tensorboard --logdir=data/logs```
