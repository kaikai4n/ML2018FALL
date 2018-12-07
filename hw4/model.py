import torch

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def _gru(self, input_dim, hidden_size):
        gru = torch.nn.GRU(
                input=input_dim,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=self._dropout,
                bidirectional=self._bidirectional,)
        return gru
    
    def _relu(self):
        return torch.nn.ReLU()

    def _softmax(self):
        return torch.nn.Softmax()

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
        # 
        self._args = args
        for key, value in self._args.items():
            eval('self._'+ key) = value
        super(RNN, self).__init__()
        if train:
            self._init_network()
        else:
            self.load(self._load_model_filename)

    def _init_network(self):
        self._embedding = torch.nn.Embedding(
                self._word_dict_len, self._embed_dim)
        self._rnn = torch.nn.Sequence(
                    self._gru(self._embed_dim, self._hidden_size),
                )
        self._linear = torch.nn.Sequence(
                    torch.nn.Linear(self._hidden_size, 32),
                    self._relu(),
                    torch.nn.Linear(32, 2),
                    self._softmax(),
                )

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sentence(
                x_embed, x_length, batch_first=True)
        _, hidden = self._rnn(x_packed, None)
        pred = self._linear(hidden)
        return pred
