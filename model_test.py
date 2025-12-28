import torch
import numpy as np
import matplotlib.pyplot as plt

import joblib

from model_train import model
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler.save')
model.load_state_dict(torch.load('seq2seq_model.pth'))

X_test = np.load('X_test_array.npy')
y_test = np.load('y_test_array.npy')

X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32)

model.eval()
with torch.inference_mode():
        X_test_tensor = X_test_tensor.to(device)
        output = model(X_test_tensor, y = None, teacher_forcing_ratio = 0)

preds_transformed = output.cpu().numpy().reshape(-1,1)
y_test_transformed = y_test_tensor.cpu().numpy().reshape(-1,1)

preds = scaler.inverse_transform(preds_transformed)
y_test = scaler.inverse_transform(y_test_transformed)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

for i in range(5):
    plt.figure()
    start = i*10
    end = (i+1)*10

    plt.plot(y_test[start:end], label = "Real", color = "red")
    plt.plot(preds[start:end], label = "Pred", color = "green")
    plt.title("Daily Prediction")
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()