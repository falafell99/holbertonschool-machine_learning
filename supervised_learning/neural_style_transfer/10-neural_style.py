#!/usr/bin/env python3
"""Module for Neural Style Transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Class to perform tasks for neural style transfer."""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Class constructor for NST."""
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image to max 512 pixels and [0, 1] range."""
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w)
        new_shape = (int(h * scale), int(w * scale))

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, new_shape, method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """Creates the model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        inputs = tf.keras.Input(shape=(None, None, 3), name='input_1')
        x = inputs
        outputs_dict = {}

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)(x)
            else:
                x = layer(x)

            outputs_dict[layer.name] = x

        outputs = [outputs_dict[name] for name in self.style_layers]
        outputs.append(outputs_dict[self.content_layer])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.trainable = False
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of an input tensor."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost."""
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0)

        style_outputs = self.model(preprocessed_style)
        content_outputs = self.model(preprocessed_content)

        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return cost

    def style_cost(self, style_outputs):
        """Calculates the overall style cost for generated image."""
        length = len(self.style_layers)
        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))

        weight = 1.0 / length
        total_cost = 0.0

        for i in range(length):
            layer_cost = self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])
            total_cost += weight * layer_cost

        return total_cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image."""
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image."""
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) not in [3, 4]):
            raise TypeError("image must be a tensor of rank 3 or 4")

        return tf.image.total_variation(generated_image)

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image."""
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0)

        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)[0]

        J_total = ((self.alpha * J_content) +
                   (self.beta * J_style) +
                   (self.var * J_var))

        return J_total, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates the gradients for the tf.Tensor generated image."""
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image."""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None and type(step) is not int:
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError("step must be positive and less than iterations")

        if type(lr) not in [float, int]:
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if type(beta1) is not float:
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if type(beta2) is not float:
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             beta_1=beta1, beta_2=beta2)

        generated_image = tf.Variable(self.content_image)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, "
                      "style {}, var {}".format(
                          i, J_total, J_content, J_style, J_var))

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                clipped = tf.clip_by_value(generated_image, 0.0, 1.0)
                generated_image.assign(clipped)

        return best_image[0], best_cost
