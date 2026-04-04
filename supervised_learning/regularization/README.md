# Regularization

This project explores various regularization techniques used in machine learning to prevent overfitting and improve generalization.

## Tasks
| Task | File | Description |
| **2. L2 Reg Cost (Keras)** | `2-l2_reg_cost.py` | Calculates the total cost for each layer using Keras model losses. |
| **1. Gradient Descent with L2** | `1-l2_reg_gradient_descent.py` | Implementation of backprop with L2 weight decay. |
| --- | --- | --- |
| **0. L2 Regularization Cost** | `0-l2_reg_cost.py` | Calculates the cost of a NN with L2 regularization. |

## Formulas
### L2 Regularization Cost
$$J_{L2} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$
