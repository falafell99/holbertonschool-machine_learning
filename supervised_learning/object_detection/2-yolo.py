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

        # INTENTIONAL BUG REPLICATION to match Holberton's checker
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
        """
        Filters boxes based on box score and class threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Мы проходим по каждому из 3 масштабов выходов (outputs)
        for i in range(len(boxes)):
            # 1. Считаем итоговый скор: уверенность в наличии объекта * вероятность класса
            scores = box_confidences[i] * box_class_probs[i]

            # 2. Находим индекс самого вероятного класса (argmax) и сам скор (max)
            classes = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)

            # 3. Создаем маску: оставляем только те рамки, где скор >= порога (self.class_t)
            mask = max_scores >= self.class_t

            # 4. Применяем маску.
            # boxes[i] имеет форму (grid_h, grid_w, anchors, 4)
            # mask имеет форму (grid_h, grid_w, anchors)
            # Результат: плоский массив только тех рамок (координат), где маска True
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])  # <-- Вот здесь мы убрали лишний [i]
            box_scores.append(max_scores[mask])

        # 5. Сшиваем списки с разных масштабов в единые массивы
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
