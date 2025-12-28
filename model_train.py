import torch
import torch.nn as nn

import numpy as np

from seq2seq_model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = np.load('X_train_array.npy')
X_test = np.load('X_test_array.npy')
y_train = np.load('y_train_array.npy')
y_test = np.load('y_test_array.npy')

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32)

INPUT_SIZE = 1
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 32

encoder_model = Encoder(input_size = INPUT_SIZE, 
                        hidden_dim = HIDDEN_DIM, 
                        num_layers = NUM_LAYERS)

decoder_model = Decoder(output_size = OUTPUT_SIZE, 
                        hidden_dim = HIDDEN_DIM, 
                        num_layers = NUM_LAYERS)

model = Seq2Seq(encoder = encoder_model,
                decoder = decoder_model,
                output_window = 10,
                device = device).to(device)

train_set = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)


train_dataLoader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)

best_loss = float("inf")
EPOCHS = 50
model.train()

for epoch in range(EPOCHS):
    train_loss = 0
    for X_batch, y_batch in train_dataLoader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X = X_batch, y = y_batch)
        loss = loss_fn(output, y_batch)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_dataLoader)
    print(f"Epoch: {epoch}/{EPOCHS} | Train Loss: {train_loss}")

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "seq2seq_model.pth")