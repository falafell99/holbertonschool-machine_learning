#!/usr/bin/env python3
"""Module to initialize a Yolo object for object detection."""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Args:
            model_path: path to where a Darknet Keras model is stored.
            classes_path: path to where the list of class names used
                          for the Darknet model can be found.
            class_t: float representing the box score threshold for
                     the initial filtering step.
            nms_t: float representing the IOU threshold for non-max
                   suppression.
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all of the anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes predictions from the Darknet model.

        Args:
            outputs: list of numpy.ndarrays containing predictions.
            image_size: numpy.ndarray containing the image's original size
                        [image_height, image_width].

        Returns:
            A tuple of (boxes, box_confidences, box_class_probs).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size
        
        # Dimensions the model was trained on (e.g., 416x416)
        model_h = self.model.input.shape[1]
        model_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # 1. Create grid of offsets (c_x, c_y)
            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            cy = np.arange(grid_height).reshape(grid_height, 1)
            cy = np.repeat(cy, grid_width, axis=1)

            grid = np.stack((cx, cy), axis=-1)
            grid = np.expand_dims(grid, axis=2)

            # 2. Extract raw coordinates
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            # 3. Calculate Center Coordinates (b_x, b_y)
            # Apply sigmoid and add grid offsets
            b_xy = (1 / (1 + np.exp(-t_xy))) + grid
            
            # Normalize centers against the grid size 
            # to get relative positions (0 to 1)
            b_xy /= [grid_width, grid_height]

            # 4. Calculate Dimensions (b_w, b_h)
            # Anchors are given relative to the model input size (416x416)
            anchors_xy = self.anchors[i]
            b_wh = np.exp(t_wh) * anchors_xy
            
            # Normalize dimensions against the model input size (0 to 1)
            b_wh /= [model_w, model_h]

            # 5. Convert to Corner Coordinates (x1, y1, x2, y2)
            # relative to the normalized space
            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)

            # 6. Scale back up to the ORIGINAL image dimensions
            box[..., [0, 2]] *= image_width
            box[..., [1, 3]] *= image_height

            boxes.append(box)

            # Extract confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
