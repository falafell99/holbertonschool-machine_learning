# Object Detection with YOLOv3

Implementation of the YOLO (You Only Look Once) v3 algorithm from scratch using TensorFlow/Keras. 

## Structure
* `0-yolo.py` - Initialization of the YOLO v3 model architecture, loading pre-trained weights, anchor boxes, and class definitions.

## Configuration
The architecture utilizes the Darknet backbone to output feature maps at three different scales, making it highly robust for detecting both large and small objects in real-time environments.
