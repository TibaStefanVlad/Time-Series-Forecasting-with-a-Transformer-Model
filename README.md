# Time Series Forecasting with a Transformer Model

## Overview

This project demonstrates how a Transformer neural network can be used for time series forecasting. The implementation is written in Python using PyTorch and is trained on the well-known AirPassengers dataset, which contains the monthly number of international airline passengers between 1949 and 1960.

The main objective of the project is to show the complete forecasting pipeline, starting from data preprocessing and sequence generation, continuing with model training, and ending with performance evaluation and visualization of the results.

## Dataset

The project uses the AirPassengers dataset provided by the Darts library. Before training, the passenger values are normalized using `StandardScaler` to improve the stability of the learning process.

The data is then transformed into supervised learning samples by creating sliding windows. Each input sequence contains 24 consecutive observations, while the model is trained to predict the following 6 time steps.

## Model Architecture

The forecasting model is based on the Transformer Encoder architecture. The input values are first projected into a higher-dimensional embedding space using a linear layer. Learnable positional embeddings are added so that the model can distinguish the position of each observation within the sequence.

The embedded sequences are processed by two Transformer Encoder layers with four attention heads. Finally, the representation of the last encoder time step is passed through a fully connected layer that produces the six predicted future values.

## Training

The model is trained using the Adam optimizer with a learning rate of 0.001 and the Mean Squared Error (MSE) loss function. Training is performed for 50 epochs with a batch size of 16.

During each epoch, the model processes every batch, computes the prediction error, updates its parameters through backpropagation, and records the average training loss. The evolution of the loss is stored and later visualized to observe how the model learns over time.

## Evaluation

Once training is completed, the model is evaluated on the available sequences. The quality of the predictions is measured using three commonly used regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

The predicted values are then compared with the actual values through a line plot, making it easier to visually assess the forecasting performance.

## Visualization

The project generates two figures after training. The first one compares the predicted values with the actual observations from the dataset, allowing the forecasting accuracy to be inspected visually. The second figure illustrates the evolution of the training loss across all epochs, providing insight into the learning process.

## Technologies

The implementation is developed in Python and makes use of several popular machine learning libraries, including PyTorch, NumPy, Pandas, Matplotlib, Scikit-learn, and Darts.

## Running the Project

After installing the required dependencies, the project can be executed by running:

```bash
python transformer_forecasting.py
```

The program will train the Transformer model, print the evaluation metrics in the console, and display the generated plots.

## Conclusion

This project provides a simple but complete example of applying Transformer networks to time series forecasting. It covers every stage of the workflow, from preprocessing the data to training, evaluating, and visualizing the model's predictions. Although the implementation is intentionally lightweight, it serves as a solid starting point for experimenting with more advanced forecasting techniques and larger real-world datasets.
