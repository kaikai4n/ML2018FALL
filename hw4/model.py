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
                dropout=self._dropout,
                bidirectional=self._bidirectional,)
        return gru
    
    def _relu(self):
        return torch.nn.ReLU()

    def _softmax(self):
        return torch.nn.Softmax(dim=1)

    def save(self, filename):
        state_dict = {name:value.cpu() for name, value \
                in self.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)
    
    def load(self, filename):
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
        if train:
            self._init_network()
        else:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(
                        2*self._hidden_size if self._bidirectional\
                                else self._hidden_size, 32),
                    self._relu(),
                    torch.nn.Linear(32, 2),
                    self._softmax(),
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
