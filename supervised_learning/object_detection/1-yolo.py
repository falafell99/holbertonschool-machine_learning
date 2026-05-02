#!/usr/bin/env python3
"""Module to initialize a Yolo object for object detection."""
import tensorflow.keras as K
import numpy as np


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

        # Явное приведение к int спасает от багов tf.Dimension
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Извлечение сырых параметров
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            t_conf = output[..., 4:5]
            t_class = output[..., 5:]

            # Вычисление уверенности и вероятностей классов
            box_confidences.append(1 / (1 + np.exp(-t_conf)))
            box_class_probs.append(1 / (1 + np.exp(-t_class)))

            # Создание сетки смещений
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)
            grid = np.expand_dims(np.stack((cx, cy), axis=-1), axis=2)

            # Вычисление центров b_x, b_y (нормализация по сетке)
            b_xy = (1 / (1 + np.exp(-t_xy))) + grid
            b_xy /= [grid_w, grid_h]

            # Вычисление размеров b_w, b_h (нормализация по входу модели)
            b_wh = np.exp(t_wh) * self.anchors[i]
            b_wh /= [input_w, input_h]

            # Преобразование координат в углы (x1, y1, x2, y2)
            b_xy1 = b_xy - b_wh / 2
            b_xy2 = b_xy + b_wh / 2

            # Сборка и масштабирование под оригинальный размер изображения
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box[..., [0, 2]] *= image_w
            box[..., [1, 3]] *= image_h

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
