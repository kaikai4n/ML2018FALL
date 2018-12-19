import torch

class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self._args = args
        for key, value in self._args.items():
            exec('self._'+ key + ' = value')
    
    def _gru(self, input_dim, hidden_size):
        gru = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=self._rnn_layers,
                dropout=self._dropout,
                bidirectional=self._bidirectional,)
        return gru
    
    def _relu(self):
        return torch.nn.ReLU()

    def _softmax(self):
        return torch.nn.Softmax(dim=1)

    def _sigmoid(self):
        return torch.nn.Sigmoid()

    def save(self, filename):
        state_dict = {name:value.cpu() for name, value \
                in self.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)
    
    def load(self, filename):
        if filename is None:
            raise Exception('Error when loading model: filename not given.')
        status = torch.load(filename)
        self.load_state_dict(status['state_dict'])
    
class RNN(BaseModel):
    def __init__(self, args, train=True):
        # args is a dictionary containing required arguments:
        # load_model_filename: the filename of init parmeters
        # word_dict_len: the length of word dictionary
        # embed_dim: embedding dimension
        # hidden_size: hidden size of RNN
        # dropout: dropout rate of RNN 
        # bidirectional: RNN is bidirectional or not
        super(RNN, self).__init__(args)
        self._init_network()
        if train == False:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(self._hidden_multiply*self._hidden_size, 32),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(32, 1),
                    torch.nn.Sigmoid(),
                )

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        batch_size = x.shape[0]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_embed, x_length, batch_first=True)
        _, hidden = self._rnn(x_packed, None)
        trans_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        pred = self._linear(trans_hidden)
        return pred

class RNNWord2Vec(BaseModel):
    def __init__(self, args, train=True):
        super(RNNWord2Vec, self).__init__(args)
        self._init_network()
        if train == False:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(
                        self._hidden_multiply*self._hidden_size + self._embed_dim, 32),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(32, 1),
                    torch.nn.Sigmoid(),
                )

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        batch_size = x.shape[0]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_embed, x_length, batch_first=True)
        _, hidden = self._rnn(x_packed, None)
        #rnn_out = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        trans_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        mean_x_embed = torch.mean(x_embed, dim=1)
        linear_in = torch.cat([trans_hidden, mean_x_embed], dim=1)
        pred = self._linear(linear_in)
        return pred

class RNNWord2VecMeanPooling(BaseModel):
    def __init__(self, args, train=True):
        super(RNNWord2VecMeanPooling, self).__init__(args)
        self._init_network()
        if train == False:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(
                        self._hidden_multiply*self._hidden_size*2 + self._embed_dim, 32),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(32, 1),
                    torch.nn.Sigmoid(),
                )

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        batch_size = x.shape[0]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_embed, x_length, batch_first=True)
        rnn_out, hidden = self._rnn(x_packed, None)
        rnn_out = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        mean_rnn_out = torch.mean(rnn_out[0], dim=1)
        trans_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        mean_x_embed = torch.mean(x_embed, dim=1)
        linear_in = torch.cat([mean_rnn_out, trans_hidden, mean_x_embed], dim=1)
        pred = self._linear(linear_in)
        return pred

class DNNBOW(BaseModel):
    def __init__(self, args, train=True):
        super(DNNBOW, self).__init__(args)
        self._init_network()
        if train == False:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(self._vocabulary_size, 2048),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(2048, 256),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(256, 32),
                    torch.nn.Dropout(0.5),
                    self._relu(),
                    torch.nn.Linear(32, 1),
                    self._sigmoid(),
                )
    
    def forward(self, x):
        return self._linear(x)
