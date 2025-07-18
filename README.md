# 🤖 Practical Implementation of Artificial Neural Network (ANN)

This project demonstrates a complete workflow of building an Artificial Neural Network (ANN) from scratch using Python, TensorFlow, and Keras. The goal is to predict customer churn based on a bank’s dataset.

---

## 📁 Project Structure

- `Practical Implementation of ANN.ipynb` — Jupyter notebook containing full code for data preprocessing, model creation, training, evaluation, and visualization.
- `Churn_Modelling.csv` — Dataset used to train the model.
- `README.md` — Project overview and instructions (you’re reading it!).

---

## 📌 Problem Statement

Predict whether a customer will leave the bank (churn) based on various input features like:
- Credit score
- Geography
- Gender
- Age
- Balance
- Estimated salary, etc.

---

## 🚀 Technologies Used

| Tool / Library        | Purpose                           |
|-----------------------|-----------------------------------|
| Python (3.x)          | Core programming language         |
| TensorFlow / Keras    | Building and training the ANN     |
| Scikit-learn (sklearn)| Preprocessing, splitting data     |
| Pandas, NumPy         | Data handling and manipulation    |
| Matplotlib / Seaborn  | Visualizations (optional)         |

---

## 🧠 ANN Architecture

- **Input Layer**: 11 neurons (features from dataset)
- **Hidden Layers**: Two layers with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation for binary classification
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `Adam` with custom learning rate
- **Metrics**: Accuracy

---

## 📊 Model Training

```python
model.fit(
    X_train,
    y_train,
    validation_split=0.33,
    batch_size=10,
    epochs=1000,
    callbacks=[early_stopping]
)
```
🛑 EarlyStopping was used to prevent overfitting by monitoring validation loss.

✅ Results
Achieved ~85% accuracy on both training and validation sets.

Training stopped early once the model reached a stable performance (no overfitting observed).

Model evaluated with .evaluate() and weights extracted using .get_weights().

.

📁 Dataset Details
Dataset: Churn_Modelling.csv
Source: Banking churn prediction dataset
Shape: 10,000 rows × 14 columns

Target Variable: Exited (1 = customer left, 0 = stayed)

🛠️ Future Improvements
Add more evaluation metrics (confusion matrix, ROC AUC)

Use GridSearchCV or RandomSearch for hyperparameter tuning

Try dropout regularization or batch normalization

