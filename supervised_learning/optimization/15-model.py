#!/usr/bin/env python3
"""Module to build, train, and save a model in TensorFlow."""
import tensorflow as tf
create_mini_batches = __import__('3-mini_batch').create_mini_batches
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def model(Data, layers, activations, alpha, beta1, beta2, epsilon,
          decay_rate, decay_step):
    """
    Builds, trains, and saves a NN model using Adam, Mini-batch,
    Learning Rate Decay, and Batch Normalization.
    """
    X_train, Y_train, X_valid, Y_valid = Data
    m, nx = X_train.shape
    ny = Y_train.shape[1]

    # Строим модель
    inputs = tf.keras.Input(shape=(nx,))
    prev = inputs
    for i in range(len(layers)):
        if i < len(layers) - 1:
            prev = create_batch_norm_layer(prev, layers[i], activations[i])
        else:
            # Последний слой обычно без Batch Norm (зависит от задачи)
            init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
            prev = tf.keras.layers.Dense(layers[i], kernel_initializer=init)(prev)
            prev = activations[i](prev)

    model = tf.keras.Model(inputs=inputs, outputs=prev)

    # Оптимизатор с затуханием
    lr_schedule = learning_rate_decay(alpha, decay_rate, decay_step)
    optimizer = create_Adam_op(lr_schedule, beta1, beta2, epsilon)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Цикл обучения (5 эпох по условию Holberton)
    for epoch in range(6): # 0 to 5
        # Печать в начале каждой эпохи
        train_loss = tf.reduce_mean(loss_fn(Y_train, model(X_train)))
        train_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_train, model(X_train)))
        valid_loss = tf.reduce_mean(loss_fn(Y_valid, model(X_valid)))
        valid_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_valid, model(X_valid)))

        print(f"After {epoch} epochs:")
        print(f"\tTraining Cost: {train_loss}")
        print(f"\tTraining Accuracy: {train_acc}")
        print(f"\tValidation Cost: {valid_loss}")
        print(f"\tValidation Accuracy: {valid_acc}")

        if epoch < 5:
            batches = create_mini_batches(X_train, Y_train, 32)
            for step, (X_batch, Y_batch) in enumerate(batches):
                with tf.GradientTape() as tape:
                    preds = model(X_batch)
                    loss = loss_fn(Y_batch, preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if (step + 1) % 100 == 0:
                    step_acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_batch, preds))
                    print(f"\tStep {step + 1}:")
                    print(f"\t\tCost: {loss}")
                    print(f"\t\tAccuracy: {step_acc}")

    model.save('model.h5')
    return model
