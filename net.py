import torch
from torch import nn
import torch.nn.functional as F

## Reduce the structure of the old model
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, encoder_outputs):
        # MultiHeadAttention requires input as [sequence len, batch size, hidden_dim]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_output, attn_output_weights = self.attention(encoder_outputs, encoder_outputs, encoder_outputs)
        outputs = attn_output.transpose(0, 1)  
        weights = attn_output_weights.transpose(0, 1)  # Transpose again to get back to [batch size, sequence len, sequence len]
        return outputs, weights


class RNNNet(nn.Module):
    """
    Define an RNN network.
    """
    def __init__(self, cat_feat, cat_mapping, embed_dim_mapping, input_dim=51, hidden_dim=100, output_dim=1, n_layers=2, drop_prob=0.2, bidirectional=True, num_heads=8, activation_num=0):
        super(RNNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cat_features = cat_feat
        self.dropout = nn.Dropout(drop_prob)
        self.bidirect = bidirectional
        self.num_heads = num_heads

        # Define the embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(len(unique_values), embed_dim_mapping[f])
            for f, unique_values in cat_mapping.items()
            })
        
        # Update the input dimension of the RNN based on the dimensions of the embeddings
        input_dim += sum(embed_dim_mapping.values()) - len(cat_mapping)

        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, 
                            dropout=drop_prob, bidirectional=self.bidirect)

        self.rnn_to_attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = SelfAttention(hidden_dim, num_heads=num_heads)


        # Add weight decay (L2 Regularization) for fully connected layers
        self.fc1 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim), dim=None)
        self.fc2 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim), dim=None)

        # Adding Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # self.fc = nn.Linear(hidden_dim, output_dim) # output_dim should now be n_future
        self.fc_mean = nn.Linear(hidden_dim, output_dim) 
        self.fc_std = nn.Linear(hidden_dim, output_dim) 

        # Define various activation functions
        self.activation_dict = {
            0: nn.ReLU(),
            1: nn.Tanh(),
            2: nn.Sigmoid(),
            3: nn.ELU(),
            4: nn.SiLU(),
            5: nn.LeakyReLU()
        }
        self.activation = self.activation_dict[activation_num]

    def forward(self, x, cat_x, h):
        x_embed = [self.embeddings[f](cat_x[:, :, i]) for i, f in enumerate(self.cat_features)]
        x_embed = torch.cat(x_embed, 2)
        x = torch.cat((x, x_embed), 2)

        rnn_out, h = self.rnn(x, h)
        
        if self.bidirect:
            rnn_out = self.rnn_to_attention(rnn_out)
        skip_rnn_out = rnn_out
        out, _ = self.attention(rnn_out)

        # skip connection
        out = out + skip_rnn_out  # Adding skip connection
        
        # feed-forward layers
        out = self.dropout(self.fc1(out))
        out = self.dropout(self.fc2(out))

        out = torch.mean(out, dim=1)
        
        out = self.ln1(out)
        out = self.activation(out)
        out = self.ln2(out)
        out = self.activation(out)

        mean = self.fc_mean(out)
        std = self.fc_std(out)
        std = F.softplus(std)
        return mean, std, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_directions = 2 if self.rnn.bidirectional else 1
        if isinstance(self.rnn, nn.LSTM):
            hidden = (weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_())
        else:
            hidden = weight.new(self.n_layers * num_directions, batch_size, self.hidden_dim).zero_()
        return hidden 