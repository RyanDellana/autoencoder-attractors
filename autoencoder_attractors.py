from __future__ import print_function

import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.datasets import mnist
from models import build_vae_conv_model


def plot_results(models,
                 data,
                 batch_size = 128,
                 model_name = "vae_mnist",
                 cycles     = 30,
                 var_scale  = 1.0):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
        cycles (int): number of decoder-encoder cycles to move through the attractor space.
        var_scale (float): value used to scale the sampling noise of the encoder.
    """

    encoder, decoder = models
    x_test, y_test = data
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    x_scale_test = np.ones(shape=(x_test.shape[0],), dtype=np.float32)
    _, z_mean, _ = encoder.predict([x_test, x_scale_test],
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    lst_x_decoded = []
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            lst_x_decoded.append(x_decoded[0])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

    x_scale_custom = np.ones(shape=(len(lst_x_decoded),), dtype=np.float32) * var_scale
    for cycle_idx in range(cycles):
        lst_x_decoded = np.array(lst_x_decoded, dtype=np.float32).reshape((900, 28, 28, 1))
        z, z_mean, z_log_var = encoder.predict([lst_x_decoded, x_scale_custom])
        lst_x_decoded = []
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([z[i*len(grid_x) + j]])
                x_decoded = decoder.predict(z_sample)
                lst_x_decoded.append(x_decoded)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        plt.figure(figsize=(10, 10))        
        plt.imshow(figure, cmap='Greys_r')
        filename = os.path.join(model_name, "digits_over_latent_"+str(cycle_idx)+".png")
        plt.savefig(filename)
        plt.show()


if __name__ == '__main__':
    # params
    epochs     = 30
    batch_size = 100
    cycles     = 30
    var_scale  = 0.0 # try 1.0 and 2.0 for comparison. ;-)

    # load dataset
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + (28, 28, 1))
    x_scale_train = np.ones(shape=(x_train.shape[0],), dtype=np.float32)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + (28, 28, 1))
    x_scale_test = np.ones(shape=(x_test.shape[0],), dtype=np.float32)

    # build model
    vae, encoder, decoder = build_vae_conv_model()

    # train model
    if not os.path.exists('vae_cnn_mnist.h5'):
        vae.fit([x_train, x_scale_train],
                shuffle    = True,
                epochs     = epochs,
                batch_size = batch_size,
                validation_data = ([x_test, x_scale_test], None))
        vae.save_weights('vae_cnn_mnist.h5')
    else:
        vae.load_weights('vae_cnn_mnist.h5')

    plot_results(models = (encoder, decoder),
                 data   = (x_test, y_test),
                 batch_size = batch_size,
                 model_name = 'vae_cnn',
                 cycles = cycles,
                 var_scale = var_scale)

