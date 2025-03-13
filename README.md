# Stock Prediction using Temporal Fusion Transformer (TFT)

This repository contains a stock prediction project using the **Temporal Fusion Transformer (TFT)** model to predict the next day's Last Traded Price (LTP) for various companies. The model is designed to integrate with a mobile application built using **Flutter**.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)


## Overview

This project aims to predict stock prices using a deep learning model called **Temporal Fusion Transformer (TFT)**. The model takes historical stock data, including features such as Open, High, Low, and LTP, to predict the next day's LTP. The predictions are then appended to the dataset for future use. Additionally, the model is designed to work in conjunction with a Flutter mobile application to provide real-time stock price predictions.

## Dataset

The dataset used for training and prediction consists of stock data for various companies. Each dataset contains the following columns:

- **Symbol**: The stock symbol or ticker.
- **Date**: The date of the stock data.
- **Open**: The opening price of the stock.
- **High**: The highest price of the stock during the trading day.
- **Low**: The lowest price of the stock during the trading day.
- **Close**: The closing price of the stock.
- **Percent Change**: The percentage change in stock price.
- **Volume**: The volume of stocks traded on that day.


## Model

The **Temporal Fusion Transformer (TFT)** model is a deep learning model designed for forecasting time-series data. In this project, we use the TFT model to predict the next day's LTP based on the historical data provided in the datasets.

The model is trained using the following process:
1. Prepare the data by cleaning and transforming it into a format suitable for the TFT model.
2. Train the model using the historical data.
3. Predict the next day's close.
4. Append the predicted value to the dataset for future predictions.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-prediction-tft.git
   cd stock-prediction-tft
   
## Results

The model gave mae at around 0.02 which is indicates model is working fine.

The model is inside nTFT folder of this repo.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
