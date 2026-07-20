#!/usr/bin/env python3
"""
Module for training the Transformer
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate schedule """
    def __init__(self, d_model, warmup_steps=4000):
        """ Constructor """
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """ Call method """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """ Calculates sparse categorical crossentropy loss ignoring padding """
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(
        real, pred, from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """ Calculates accuracy ignoring padding """
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int64))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ Creates and trains a transformer model """
    dataset = Dataset(batch_size, max_len)

    # Add 2 for start and end tokens
    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        """ Single training step executed efficiently as a Graph """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = \
            create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True, enc_padding_mask,
                                      combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print("Epoch {}, Batch {}: Loss {}, Accuracy {}".format(
                    epoch + 1, batch, float(train_loss.result()),
                    float(train_accuracy.result())))

        print("Epoch {}: Loss {}, Accuracy {}".format(
            epoch + 1, float(train_loss.result()),
            float(train_accuracy.result())))

    return transformer
