#!/usr/bin/env python3
"""Module that contains the WGAN_clip class."""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN with weight clipping class."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the WGAN_clip model."""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer:
        # Generator wants to maximize discriminator's output for fake images
        self.generator.loss = lambda fake: -tf.math.reduce_mean(fake)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        # Discriminator wants to maximize real preds and minimize fake preds
        self.discriminator.loss = lambda real, fake: (
            tf.math.reduce_mean(fake) - tf.math.reduce_mean(real)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Generator of fake samples."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size),
                              training=training)

    def get_real_sample(self, size=None):
        """Generator of real samples."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Training step for the WGAN."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=False)

                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            discr_grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_grads, self.discriminator.trainable_variables))

            # clip the weights (of the discriminator) between -1 and 1
            for weight in self.discriminator.trainable_variables:
                weight.assign(tf.clip_by_value(weight, -1.0, 1.0))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        gen_grads = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
