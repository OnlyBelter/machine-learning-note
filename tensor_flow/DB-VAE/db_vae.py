"""
MIT 6.S191, lab2
https://github.com/aamini/introtodeeplearning/blob/master/lab2/Part2_Debiasing.ipynb
http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf
"""

import IPython
import tensorflow as tf
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mitdeeplearning as mdl


def make_standard_classifier(n_outputs=1, n_filters=12):
    """Function to define a standard CNN model
    :param n_outputs: the number of units in the last layer
    :param n_filters: base number of convolutional filters
    """
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

    model = tf.keras.Sequential([
        Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        Dense(512),
        Dense(n_outputs, activation=None),
    ])
    return model


@tf.function
def standard_train_step(x, y):
    with tf.GradientTape() as tape:
        # feed the images into the model
        logits = standard_classifier(x)
        # Compute the loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    # Backpropagation
    grads = tape.gradient(loss, standard_classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
    return loss


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    """ Function to calculate VAE loss given:
          an input x,
          reconstructed output x_recon,
          encoded means mu,
          encoded log of standard deviation logsigma,
          weight parameter for the latent loss kl_weight
    """
    # Define the latent loss. Note this is given in the equation for L_{KL} in the text block, and measures how closely
    # the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=1)

    # Define the reconstruction loss as the mean absolute pixel-wise
    # difference between the input and reconstruction. Hint: you'll need to
    # use tf.reduce_mean, and supply an axis argument which specifies which
    # dimensions to reduce over. For example, reconstruction loss needs to average
    # over the height, width, and channel image dimensions.
    # https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
    reconstruction_loss = tf.reduce_mean(tf.abs(tf.math.subtract(x, x_recon)), axis=(1, 2, 3))

    # Define the VAE loss. Note this is given in the equation for L_{VAE}
    # in the text block directly above
    vae_loss = kl_weight * latent_loss + reconstruction_loss

    return vae_loss


def sampling(z_mean, z_logsigma):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
    # Returns
        z (tensor): sampled latent vector
    """
    # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
    batch, latent_dim = z_mean.shape  # 32 x 100
    epsilon = tf.random.normal(shape=(batch, latent_dim))

    # Define the reparameterization computation!
    # Note the equation is given in the text block immediately above.
    z = z_mean + tf.exp(0.5 * z_logsigma) * epsilon  # changed a little from the center of each sample
    return z


def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    """Loss function for DB-VAE.
    # Arguments
        x: true input x
        x_pred: reconstructed x
        y: true label (face or not face)
        y_logit: predicted labels
        mu: mean of latent distribution (Q(z|X))
        logsigma: log of standard deviation of latent distribution (Q(z|X))
    # Returns
        total_loss: DB-VAE total loss
        classification_loss = DB-VAE classification loss
    """
    # call the relevant function to obtain VAE loss
    vae_loss = vae_loss_function(x=x, x_recon=x_pred, mu=mu, logsigma=logsigma)

    # define the classification loss using sigmoid_cross_entropy
    # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)

    # Use the training data labels to create variable face_indicator:
    #   indicator that reflects which training data are images of faces
    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)

    # define the DB-VAE total loss! Use tf.reduce_mean to average over all
    # samples
    total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)

    return total_loss, classification_loss


class DB_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder. Recall that we have
        # `latent_dim` latent variables, as well as a supervised output for the
        # classification (index 0).
        num_encoder_dims = 2 * self.latent_dim + 1  # 2 * 100 + 1

        self.encoder = make_standard_classifier(num_encoder_dims)
        self.decoder = make_face_decoder_network()

    # function to feed images into encoder, encode the latent space, and output
    #   classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return y_logit, z_mean, z_logsigma

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        # call the sampling function defined above
        z = sampling(z_mean=z_mean, z_logsigma=z_logsigma)
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        # use the decoder to output the reconstruction
        reconstruction = self.decoder(z)  # how to pass z???
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # reparameterization
        z = self.reparameterize(z_mean=z_mean, z_logsigma=z_logsigma)

        # reconstruction
        recon = self.decode(z)
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit


def make_face_decoder_network(n_filters=12):
    # Functionally define the different layer types we will use, the decoder portion of the DB-VAE
    Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
    Reshape = tf.keras.layers.Reshape

    # Build the decoder network using the Sequential API
    decoder = tf.keras.Sequential([
        # Transform to pre-convolutional generation
        Dense(units=4 * 4 * 6 * n_filters),  # 4x4 feature maps (with 6N occurances)
        Reshape(target_shape=(4, 4, 6 * n_filters)),

        # Upscaling convolutions (inverse of encoder)
        Conv2DTranspose(filters=4 * n_filters, kernel_size=3, strides=2),
        Conv2DTranspose(filters=2 * n_filters, kernel_size=3, strides=2),
        Conv2DTranspose(filters=1 * n_filters, kernel_size=5, strides=2),
        Conv2DTranspose(filters=3, kernel_size=5, strides=2),
    ])

    return decoder


def get_latent_mu(images, dbvae, batch_size=1024):
    # Function to return the means for an input image batch
    N = images.shape[0]
    mu = np.zeros((N, latent_dim))
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind + batch_size, N + 1)
        batch = (images[start_ind:end_ind]).astype(np.float32) / 255.
        _, batch_mu, _ = dbvae.encode(batch)
        mu[start_ind:end_ind] = batch_mu
    return mu


def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001):
    """Function that recomputes the sampling probabilities for images within a batch
        based on how they distribute across the training data
        Resampling algorithm for DB-VAE
          """
    print("Recomputing the sampling probabilities")

    # run the input batch and get the latent variable means, dim[1:101], 54957 x 100
    mu = get_latent_mu(images=images, dbvae=dbvae)

    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])
    # find the highest density across all latent dimensions for each sample
    # consider the distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data sample falls in
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        # call the digitize function to find which bins in the latent distribution
        #    every data sample falls in to
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        bin_idx = np.digitize(latent_distribution, bin_edges)

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function, assign prob for each sample from 10 values, lower density becomes bigger number
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])

        # normalize all probabilities, the group of samples which has lower density will be assigned higher prob, cool!
        p = p / np.sum(p)

        # update sampling probabilities by considering whether the newly
        #     computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(p, training_sample_p)

    # final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p  # the distribution of all positive samples


@tf.function
def debiasing_train_step(x, y, optimizer, dbvae):
    # To define the training operation, we will use tf.function which is a powerful tool
    #   that lets us turn a Python function into a TensorFlow computation graph.
    with tf.GradientTape() as tape:
        # Feed input x into dbvae. Note that this is using the DB_VAE call function!
        y_logit, z_mean, z_logsigma, x_recon = dbvae(x)

        '''TODO: call the DB_VAE loss function to compute the loss'''
        loss, class_loss = debiasing_loss_function(x=x, x_pred=x_recon, mu=z_mean, logsigma=z_logsigma, y=y,
                                                   y_logit=y_logit)

    '''TODO: use the GradientTape.gradient method to compute the gradients.
       Hint: this is with respect to the trainable_variables of the dbvae.'''
    grads = tape.gradient(loss, dbvae.trainable_variables)

    # apply gradients to variables
    optimizer.apply_gradients(zip(grads, dbvae.trainable_variables))
    return loss


if __name__ == '__main__':
    # datasets
    path_to_training_data = tf.keras.utils.get_file('../../downloads/train_face.h5',
                                                    'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
    loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data)
    number_of_training_examples = loader.get_train_size()
    (images, labels) = loader.get_batch(100)
    ### Define the CNN model ###
    standard_classifier = make_standard_classifier()
    ### Train the standard CNN ###

    # Training hyperparameters
    batch_size = 32
    num_epochs = 2  # keep small to run faster
    learning_rate = 5e-4
    latent_dim = 100  # number of latent variables

    optimizer = tf.keras.optimizers.Adam(learning_rate)  # define our optimizer
    loss_history = mdl.util.LossHistory(smoothing_factor=0.99)  # to record loss evolution
    plotter = mdl.util.PeriodicPlotter(sec=2, scale='semilogy')
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists

    # The training loop!
    for epoch in range(num_epochs):
        for idx in tqdm(range(loader.get_train_size() // batch_size)):
            # Grab a batch of training data and propagate through the network
            x, y = loader.get_batch(batch_size)
            loss = standard_train_step(x, y)

            # Record the loss and plot the evolution of the loss as a function of training
            loss_history.append(loss.numpy().mean())
            plotter.plot(loss_history.get())  # plot loss

    # Evaluation of standard CNN
    (batch_x, batch_y) = loader.get_batch(5000)
    y_pred_standard = tf.round(tf.nn.sigmoid(standard_classifier.predict(batch_x)))
    acc_standard = tf.reduce_mean(tf.cast(tf.equal(batch_y, y_pred_standard), tf.float32))

    print(f'Standard CNN accuracy on (potentially biased) training set: {acc_standard.numpy():.4f}')

    ### Load test dataset and plot examples ###

    test_faces = mdl.lab2.get_test_faces()
    keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
    for group, key in zip(test_faces, keys):
        plt.figure(figsize=(5, 5))
        plt.imshow(np.hstack(group))
        plt.title(key, fontsize=15)
    plt.close()

    ### Evaluate the standard CNN on the test data ###

    standard_classifier_logits = [standard_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
    standard_classifier_probs = tf.squeeze(tf.sigmoid(standard_classifier_logits))  # 4 by 5

    # Plot the prediction accuracies per demographic
    xx = range(len(keys))
    yy = standard_classifier_probs.numpy().mean(1)  # mean probability
    plt.bar(xx, yy)
    plt.xticks(xx, keys)
    plt.ylim(max(0, yy.min() - yy.ptp() / 2.), yy.max() + yy.ptp() / 2.)
    plt.title("Standard classifier predictions")

    ### Defining the VAE loss function ###
    ### VAE Reparameterization ###
    ### Defining and creating the DB-VAE ###

    ### Training the DB-VAE ###

    # Hyperparameters
    batch_size = 32
    learning_rate = 5e-4
    latent_dim = 100

    # DB-VAE needs slightly more epochs to train since its more complex than
    # the standard classifier so we use 6 instead of 2
    num_epochs = 6

    # instantiate a new DB-VAE model and optimizer
    # dbvae = DB_VAE(100)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # get training faces from data loader
    all_faces = loader.get_all_train_faces()  # only positive samples, 54957 x 64 x 64 x 3
    dbvae = DB_VAE(latent_dim=latent_dim)

    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    # The training loop -- outer loop iterates over the number of epochs
    for i in range(num_epochs):

        IPython.display.clear_output(wait=True)
        print("Starting epoch {}/{}".format(i + 1, num_epochs))

        # Recompute data sampling proabilities
        '''recompute the sampling probabilities for debiasing'''
        # the probability of being a face for all images
        p_faces = get_training_sample_probabilities(images=all_faces, dbvae=dbvae)

        # get a batch of training data and compute the training step
        for j in tqdm(range(loader.get_train_size() // batch_size)):
            # load a batch of data
            (x, y) = loader.get_batch(batch_size, p_pos=p_faces)  # also got some negative samples
            # loss optimization
            loss = debiasing_train_step(x, y, optimizer=optimizer, dbvae=dbvae)

            # plot the progress every 200 steps
            if j % 500 == 0:
                mdl.util.plot_sample(x, y, dbvae)

    dbvae_logits = [dbvae.predict(np.array(x, dtype=np.float32)) for x in test_faces]
    dbvae_probs = tf.squeeze(tf.sigmoid(dbvae_logits))

    xx = np.arange(len(keys))
    plt.bar(xx, standard_classifier_probs.numpy().mean(1), width=0.2, label="Standard CNN")
    plt.bar(xx + 0.2, dbvae_probs.numpy().mean(1), width=0.2, label="DB-VAE")
    plt.xticks(xx, keys)
    plt.title("Network predictions on test dataset")
    plt.ylabel("Probability")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
