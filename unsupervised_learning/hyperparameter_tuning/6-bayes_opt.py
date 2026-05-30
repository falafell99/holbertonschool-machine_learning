#!/usr/bin/env python3
"""
Optimizes a machine learning model using GPyOpt.
"""
import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def build_and_train_model(x):
    """
    Builds, trains, and evaluates the Keras model.
    """
    # 1. Извлечение гиперпараметров из массива GPyOpt
    learning_rate = float(x[:, 0])
    units = int(x[:, 1])
    dropout_rate = float(x[:, 2])
    l2_weight = float(x[:, 3])
    batch_size = int(x[:, 4])

    # Загрузка и подготовка данных (Breast Cancer Dataset)
    data = load_breast_cancer()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 2. Построение архитектуры модели
    model = Sequential([
        Dense(units, activation='relu',
              kernel_regularizer=l2(l2_weight),
              input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 3. Настройка коллбэков (Early Stopping и Checkpoints)
    filename = (f"checkpoint_lr_{learning_rate:.4f}_units_{units}_"
                f"drop_{dropout_rate:.2f}_l2_{l2_weight:.4f}_"
                f"batch_{batch_size}.h5")

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    checkpoint = ModelCheckpoint(filepath=filename,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=0)

    # 4. Обучение
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[early_stop, checkpoint],
                        verbose=0)

    # Мы минимизируем Validation Loss
    val_loss = np.min(history.history['val_loss'])

    # Очистка сессии для освобождения памяти
    tf.keras.backend.clear_session()

    return val_loss


def objective_function(x):
    """
    Wrapper for GPyOpt to consume the 2D array and return 2D loss.
    """
    loss = build_and_train_model(x)
    return np.array([[loss]])


if __name__ == '__main__':
    # Определяем 5 гиперпараметров и их границы
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
        {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    # Инициализация GPyOpt (Bayesian Optimization)
    myBopt = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        maximize=False
    )

    # Запуск оптимизации (максимум 30 итераций)
    myBopt.run_optimization(max_iter=30)

    # Генерация графиков конвергенции
    myBopt.plot_convergence(filename='convergence.png')

    # Сохранение отчета
    with open('bayes_opt.txt', 'w') as f:
        f.write("Bayesian Optimization Report\n")
        f.write("============================\n")
        f.write("Optimal Hyperparameters:\n")
        f.write(f"- Learning Rate: {myBopt.x_opt[0]:.5f}\n")
        f.write(f"- Units: {int(myBopt.x_opt[1])}\n")
        f.write(f"- Dropout Rate: {myBopt.x_opt[2]:.5f}\n")
        f.write(f"- L2 Weight: {myBopt.x_opt[3]:.5f}\n")
        f.write(f"- Batch Size: {int(myBopt.x_opt[4])}\n")
        f.write(f"\nBest Validation Loss: {myBopt.fx_opt[0]:.5f}\n")
