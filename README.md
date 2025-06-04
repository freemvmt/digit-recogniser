# Digit recogniser

> v1 is *live* at [mnist.commune.london](https://mnist.commune.london/), or directly at [157.180.89.245](http://157.180.89.245/).

This repo demonstrates the use of a convolutional neural network (CNN) to train a model on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset of handwritten single digits.

We further make the model available via a Streamlit frontend (behind a caddy-managed reverse proxy), allowing the user to sketch a digit themselves and see if the model can identify it correctly. They can then submit their effort (along with a truth label) to a database in the backend, with an eye to refining the model later on.


## Training

Running the training pipeline is very simple: cd into `backend/`, run `uv sync`, activate the virtual environment, and run `python train.py`.

For more verbose logging and other DX bits, `export DEBUG_MODE=1` in the shell beforehand.

You may want to adjust the hyperparameters at the top of the file first. The script will dump the resulting weights in `backend/models/`, and plots of training/validation loss and validation accuracy in `backend/plots/` for your examination (e.g. to consider whether the model may be under- or over-fitted).


## Docker

First replicate the `.env.example` as `.env`, supplying a password for the db you will initialise (see `db/init.sql`).

To build and run the Docker swarm locally, run `docker compose up -d`.

Some utility scripts are supplied for ease: `up.sh`, or `upforce.sh` to rebuild images and recreate all containers.

You should then find the Streamlit app at [https://localhost](https://localhost) (browser will warn, since we are self-certifying via caddy, but just click through). You can also inspect the db via something like [pgAdmin](https://www.pgadmin.org/).

Finally, run `docker compose down` to kill the containers.


### Logs

The `-d` (or `--detach`) flag does means containers will be backgrounded, so logs will not be emitted in the shell. If you want to tail these after the fact, run something like `docker compose logs -f --tail 20`.


## Deployment

Deployment should be as simple as...

- Grabbing a box (e.g. on [Hetzner](https://www.hetzner.com/)) with a fixed IP
- Running through the commands in `scripts/` (these are intended only as a steer)
- Cloning the [repo](https://github.com/freemvmt/digit-recogniser)
- Spinning up the Docker stack as above, on the server
- Hitting the [IP address](http://157.180.89.245) associated with the server!


### Troubleshooting

1. Did you forget to define your env vars? On your server, at root of the repo, do something like...

```
cp .env.example .env
vim .env
```

And choose some reasonable value for `POSTGRES_PASSWORD`!


## Resources and reference material

- Relevant PyTorch tutorials on [neural networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) and [training a classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- A [StackExchange answer](https://stats.stackexchange.com/questions/376312/mnist-digit-recognition-what-is-the-best-we-can-get-with-a-fully-connected-nn-o) about high accuracy MNIST models without convolution
- A [Kaggle notebook](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook) on using a CNN for MNIST
- A handy [visualisation](https://adamharley.com/nn_vis/cnn/2d.html) of a CNN as applied to the MNIST dataset
- A very thorough [walkthrough](https://medium.com/data-science-collective/implementing-cnn-in-pytorch-testing-on-mnist-99-26-test-accuracy-5c63876c6ac8) on building a CNN for MNIST using PyTorch
- Some notes on [recognising overfitting](https://datahacker.rs/018-pytorch-popular-techniques-to-prevent-the-overfitting-in-a-neural-networks/)
- The Caddy docs also proved [very](https://caddyserver.com/docs/automatic-https#local-https) [handy](https://caddyserver.com/docs/caddyfile/directives/tls)
