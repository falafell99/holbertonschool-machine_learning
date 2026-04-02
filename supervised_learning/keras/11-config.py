#!/usr/bin/env python3
"""Module to save and load model configuration in JSON format."""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
    network: the model whose configuration should be saved.
    filename: the path to the file.
    Returns: None.
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.
    filename: the path to the JSON file.
    Returns: the loaded model.
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
