import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True) 
    
    def forward(self, x):
        encoder_output, (hidden, cell) = self.lstm(x)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size = output_size, hidden_size = hidden_dim, num_layers= num_layers, batch_first = True)
        self.fc = nn.Linear(in_features = hidden_dim, out_features = output_size)
        
    def forward(self, x, encoder_hidden, encoder_cell):
        output, (decoder_hidden, decoder_cell) = self.lstm(x, (encoder_hidden, encoder_cell))
        
        prediction = self.fc(output)
        return prediction, decoder_hidden, decoder_cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_window, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window
        self.device = device

    def forward(self, X, y = None, teacher_forcing_ratio = 0.5):
        outputs = []
        hidden, cell = self.encoder(X)
        decoder_input = X[:,-1:,:]
        
        for i in range(self.output_window):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, i:i+1, :]
            else:
                decoder_input = prediction
        
        return torch.cat(outputs, dim = 1)