# Linear Regression in Rust

A lightweight implementation of multiple linear regression from scratch in Rust using batch gradient descent and stochastic gradient descent (SGD).

## Dataset
Trained on `Housing.csv` to predict housing prices based on features such as area, bedrooms, bathrooms, stories, main road access, furnishing status, and amenities.

## Features
- **Feature Normalization**: Z-score normalization for numerical features and one-hot/categorical encodings.
- **Optimization**:
  - Full Batch Gradient Descent (`train`)
  - Stochastic Gradient Descent (`sgd_train`)
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - $R^2$ Coefficient of Determination score

## Running
Ensure you have Rust and Cargo installed:

```bash
cargo run
```
