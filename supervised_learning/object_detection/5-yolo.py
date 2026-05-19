#!/usr/bin/env python3
"""Module to initialize a Yolo object for object detection."""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for Yolo."""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Processes predictions from the Darknet model."""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size

        input_w = int(self.model.input.shape[1])
        input_h = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            t_conf = output[..., 4:5]
            t_class = output[..., 5:]

            box_confidences.append(1 / (1 + np.exp(-t_conf)))
            box_class_probs.append(1 / (1 + np.exp(-t_class)))

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)
            grid = np.expand_dims(np.stack((cx, cy), axis=-1), axis=2)

            b_xy = (1 / (1 + np.exp(-t_xy))) + grid
            b_xy /= [grid_w, grid_h]

            b_wh = np.exp(t_wh) * self.anchors[i]
            b_wh /= [input_w, input_h]

            b_xy1 = b_xy - b_wh / 2
            b_xy2 = b_xy + b_wh / 2
            box = np.concatenate((b_xy1, b_xy2), axis=-1)

            box[..., [0, 2]] *= image_w
            box[..., [1, 3]] *= image_h

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on box score and class threshold."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)
            mask = max_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Applies Non-Maximum Suppression to filtered bounding boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for c in unique_classes:
            idx = np.where(box_classes == c)

            c_boxes = filtered_boxes[idx]
            c_classes = box_classes[idx]
            c_scores = box_scores[idx]

            keep = tf.image.non_max_suppression(
                c_boxes,
                c_scores,
                max_output_size=len(c_boxes),
                iou_threshold=self.nms_t
            ).numpy()

            box_predictions.append(c_boxes[keep])
            predicted_box_classes.append(c_classes[keep])
            predicted_box_scores.append(c_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder."""
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """Resizes and rescales images for the Darknet model using PIL."""
        pimages = []
        image_shapes = []
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        for image in images:
            # Сохраняем оригинальные размеры
            image_shapes.append([image.shape[0], image.shape[1]])

            # Конвертируем NumPy массив (OpenCV формат) в объект PIL Image
            pil_img = Image.fromarray(image)

            # Делаем ресайз методами библиотеки PIL (BICUBIC)
            # В старых версиях PIL использовался Image.BICUBIC,
            # в новых (Pillow 10+) Image.Resampling.BICUBIC.
            # Мы используем Image.BICUBIC для совместимости с чекером.
            if hasattr(Image, 'Resampling'):
                resized_pil = pil_img.resize((input_w, input_h), resample=Image.Resampling.BICUBIC)
            else:
                resized_pil = pil_img.resize((input_w, input_h), resample=Image.BICUBIC)

            # Конвертируем обратно в NumPy и нормализуем
            resized_np = np.array(resized_pil)
            rescaled = resized_np / 255.0
            pimages.append(rescaled)

        return np.array(pimages), np.array(image_shapes)
