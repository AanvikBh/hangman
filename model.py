import torch
import torch.nn as nn
import math

class PositionEncoder(nn.Module):
    """Adds positional information to input vectors."""
    def __init__(self, model_dimension, max_sequence_len=50):
        super(PositionEncoder, self).__init__()
        pos_encoding = torch.zeros(max_sequence_len, model_dimension)
        indices = torch.arange(0, max_sequence_len, dtype=torch.float).unsqueeze(1)
        scale_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension))
        pos_encoding[:, 0::2] = torch.sin(indices * scale_term)
        pos_encoding[:, 1::2] = torch.cos(indices * scale_term)
        self.register_buffer('encoding_buffer', pos_encoding.unsqueeze(0))

    def forward(self, input_tensor):
        return input_tensor + self.encoding_buffer[:, :input_tensor.size(1)]

class FeatureNormalizer(nn.Module):
    """Implements layer normalization."""
    def __init__(self, feature_count, epsilon=1e-6):
        super(FeatureNormalizer, self).__init__()
        self.gain = nn.Parameter(torch.ones(feature_count))
        self.bias = nn.Parameter(torch.zeros(feature_count))
        self.epsilon = epsilon

    def forward(self, input_tensor):
        avg = input_tensor.mean(-1, keepdim=True)
        stdev = input_tensor.std(-1, keepdim=True)
        return self.gain * (input_tensor - avg) / (stdev + self.epsilon) + self.bias

class SkipConnection(nn.Module):
    """A residual connection followed by dropout and normalization."""
    def __init__(self, feature_size, dropout_rate):
        super(SkipConnection, self).__init__()
        self.normalizer = FeatureNormalizer(feature_size)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_tensor, sub_layer_module):
        return input_tensor + self.dropout_layer(sub_layer_module(self.normalizer(input_tensor)))

class PointwiseFeedForward(nn.Module):
    """A two-layer feed-forward network."""
    def __init__(self, model_dim, ff_dim, dropout_val=0.1):
        super(PointwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout_val)
        self.activation = nn.ReLU()

    def forward(self, tensor_in):
        return self.linear_2(self.dropout(self.activation(self.linear_1(tensor_in))))

class OutputProjection(nn.Module):
    """Projects hidden states to the vocabulary space."""
    def __init__(self, hidden_dim, vocab_cardinality):
        super(OutputProjection, self).__init__()
        self.projection_layer = nn.Linear(hidden_dim, vocab_cardinality)

    def forward(self, hidden_tensor):
        # Return raw logits, not softmax probabilities
        return self.projection_layer(hidden_tensor)

class HangmanGuesserNetwork(nn.Module):
    """The main BiLSTM network for Hangman character prediction."""
    def __init__(self, vocab_size, hidden_dim, vector_dim=100,
                 layer_count=3, dropout_prob=0.2, pretrained_vectors=None):
        super(HangmanGuesserNetwork, self).__init__()
        
        if pretrained_vectors is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_vectors), freeze=False)
        else:
            self.token_embedding = nn.Embedding(vocab_size, vector_dim)
        
        self.position_encoder = PositionEncoder(vector_dim)
        
        self.recurrent_layer = nn.LSTM(vector_dim, hidden_dim, layer_count,
                                       batch_first=True, bidirectional=True, dropout=dropout_prob)
        
        lstm_out_dim = hidden_dim * 2
        
        self.ff_layer = PointwiseFeedForward(lstm_out_dim, hidden_dim * 4, dropout_prob)
        self.residual_connector = SkipConnection(lstm_out_dim, dropout_prob)
        self.final_projection = OutputProjection(lstm_out_dim, vocab_size)

    def forward(self, token_indices, sequence_lengths):
        embedded_tokens = self.token_embedding(token_indices)
        positioned_tokens = self.position_encoder(embedded_tokens)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            positioned_tokens, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_recurrent_output, _ = self.recurrent_layer(packed_input)
        recurrent_output, _ = nn.utils.rnn.pad_packed_sequence(packed_recurrent_output, batch_first=True)
        
        ff_output = self.residual_connector(recurrent_output, self.ff_layer)
        final_output = self.final_projection(ff_output)
        
        return final_output
