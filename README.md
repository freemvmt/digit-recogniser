# Digit recogniser

This repo demonstrates the use of a convolutional neural network (CNN) to train a model on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset of handwritten single digits.

We further make the model available via a Streamlit frontend, allowing the user to sketch a digit themselves and see if the model can identify it correctly. They can then submit their effort (along with a truth label) to a database in the backend, with an eye to refining the model later on.


## Training

Run the training pipeline is very simple:

```
python src/train.py
```

For more verbose logging, `export LOG_LEVEL_DEBUG=1` in the shell before running.

You may want to adjust the hyperparameters at the top of the file first. The script will dump the resulting weights in `models/`, and plots of training/validation loss and validation accuracy in `plots/` for your examination (e.g. to consider whether the model may be under- or over-fitted).


## Resources and reference material

- Relevant PyTorch tutorials on [neural networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) and [training a classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- A [StackExchange answer](https://stats.stackexchange.com/questions/376312/mnist-digit-recognition-what-is-the-best-we-can-get-with-a-fully-connected-nn-o) about high accuracy MNIST models without convolution
- A [Kaggle notebook](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook) on using a CNN for MNIST
- A handy [visualisation](https://adamharley.com/nn_vis/cnn/2d.html) of a CNN as applied to the MNIST dataset
- A very thorough [walkthrough](https://medium.com/data-science-collective/implementing-cnn-in-pytorch-testing-on-mnist-99-26-test-accuracy-5c63876c6ac8) on building a CNN for MNIST using PyTorch
- Some notes on [recognising overfitting](https://datahacker.rs/018-pytorch-popular-techniques-to-prevent-the-overfitting-in-a-neural-networks/)
