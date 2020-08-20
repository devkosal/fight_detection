# improved on https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from transformers import GPT2Model


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

    
class CRNNEncoderDecoder(nn.Module):
    def __init__(self, num_classes):
        super(EncoderDecoder, self).__init__()
        self.num_classes = num_classes
        self.cnn_encoder = ResCNNEncoder()
        self.rnn_decoder = DecoderRNN(num_classes=self.num_classes)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.rnn_decoder(x)
        return x
## ---------------------- end of CRNN module ---------------------- ##


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        """
        d_model: number of expected features in the input (required). 
        https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html
        
        source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_layers=8, num_heads=8, h_FC_dim=128, drop_p=0.3, num_classes=2):
        super().__init__()
        self.CNN_embed_dim = CNN_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        
        self.transformer = nn.Transformer(self.CNN_embed_dim, self.num_heads, self.num_layers, self.num_layers)
        self.fc1 = nn.Linear(self.CNN_embed_dim, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        self.query_pos = nn.Parameter(torch.rand(100, CNN_embed_dim))
        self.pos_encoder = PositionalEncoding(CNN_embed_dim)
            
    def forward(self, x):
        x = self.pos_encoder(x.permute(1,0,2))
        tgt = self.query_pos.unsqueeze(1)
        ts = tgt.size()
        tgt = tgt + torch.zeros(ts[0],x.size(1),ts[-1])
        x = self.transformer(x, tgt)
        x = self.fc1(x[-1])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x
    
class TfmrEncoderDecoder(nn.Module):
    def __init__(self, num_classes=2, num_heads=6):
        super(TfmrEncoderDecoder, self).__init__()
        self.num_classes = num_classes
        self.cnn_encoder = ResCNNEncoder()
        self.tfmr_decoder = TransformerDecoder(
            num_classes=self.num_classes, num_heads=num_heads
        )

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.tfmr_decoder(x)
        return x
## ---------------------- end of Transformer mode ---------------------- ##

class GPTDecoder(nn.Module):
    def __init__(self,gpt_config,CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2):
        super().__init__()
        self.CNN_embed_dim = CNN_embed_dim
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.gpt = GPT2Model(gpt_config)
        self.fc1 = nn.Linear(self.CNN_embed_dim*2, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        self.mp = nn.AdaptiveMaxPool1d(1)
        self.ap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.gpt(inputs_embeds=x)[0] # last hidden state
        x = x.permute(0, 2, 1)
        x = torch.cat([self.mp(x), self.ap(x)], 1).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


    
    
            
        
        
        
        
        
        
        