import torch
import torch.nn.functional as F

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

class VideoRNN(BaseModel):
    def __init__(self, args, train=True):
        # args is a dictionary containing required arguments:
        # load_model_filename: the filename of init parmeters
        # word_dict_len: the length of word dictionary
        # embed_dim: embedding dimension
        # hidden_size: hidden size of RNN
        # dropout: dropout rate of RNN 
        # bidirectional: RNN is bidirectional or not
        # encode_rnn_hidden: use mean of all RNN hidden states
        # encode_all: use mean of all input features
        # encode_dim: output dimension of encoding
        # common_space_dim: output dimension of common space
        # dist_fn: distance function
        super(VideoRNN, self).__init__(args)
        self._init_network(args)

    def _init_network(self, args):
        self._rnn = self._gru(4096, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2

        self._in_space_dim = self._hidden_size * self._hidden_multiply
        self._out_space_dim = self._in_space_dim
        if self._encode_rnn_hidden:
            self._in_space_dim += self._hidden_size * self._hidden_multiply
            self._out_space_dim += self._hidden_size * self._hidden_multiply
        if self._encode_all:
            self._in_space_dim += 4096
            self._out_space_dim = self._encode_dim
            self._linear = torch.nn.Sequential(
                    torch.nn.Linear(self._in_space_dim, self._out_space_dim),
                    torch.nn.BatchNorm1d(self._out_space_dim))

    def forward(self, x):
        # x is a tensor of video features: [batch, 80, 4096]
        batch_size = x.shape[0]
        rnn_out, hidden = self._rnn(x, None)
        output = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        if self._encode_rnn_hidden:
            mean_rnn_out = torch.mean(rnn_out, dim=1)
            output = torch.cat([mean_rnn_out, output], dim=1)
        if self._encode_all:
            mean_x = torch.mean(x, dim=1)
            output = torch.cat([mean_x, output], dim=1)
            output = self._linear(output)
        return output

class CaptionRNN(BaseModel):
    def __init__(self, args, train=True):
        # args is a dictionary containing required arguments:
        # load_model_filename: the filename of init parmeters
        # word_dict_len: the length of word dictionary
        # embed_dim: embedding dimension
        # hidden_size: hidden size of RNN
        # dropout: dropout rate of RNN 
        # bidirectional: RNN is bidirectional or not
        # encode_rnn_hidden: use mean of all RNN hidden states
        # encode_all: use mean of all input features
        # encode_dim: output dimension of encoding
        # common_space_dim: output dimension of common space
        # dist_fn: distance function
        super(CaptionRNN, self).__init__(args)
        self._init_network(args)

    def _init_network(self, args):
        self._embedding = torch.nn.Embedding(
                self._vocabulary_size, self._embed_dim)
        self._rnn = self._gru(self._embed_dim, self._hidden_size)
        self._hidden_multiply = self._rnn_layers
        if self._bidirectional:
            self._hidden_multiply *= 2

        self._in_space_dim = self._hidden_size * self._hidden_multiply
        self._out_space_dim = self._in_space_dim
        if self._encode_rnn_hidden:
            self._in_space_dim += self._hidden_size * self._hidden_multiply
            self._out_space_dim += self._hidden_size * self._hidden_multiply
        if self._encode_all:
            self._in_space_dim += 128
            self._out_space_dim = self._encode_dim
            self._linear = torch.nn.Sequential(
                    torch.nn.Linear(self._in_space_dim, self._out_space_dim),
                    torch.nn.BatchNorm1d(self._out_space_dim))

    def forward(self, x, x_length):
        # x is a tensor of sentence: [batch, max_sentence_length]
        # x_length is the length number of each sentence: [batch]
        batch_size = x.shape[0]
        x_embed = self._embedding(x)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_embed, x_length, batch_first=True)
        rnn_out, hidden = self._rnn(x_packed, None)
        output = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        if self._encode_rnn_hidden:
            rnn_out = torch.nn.utils.rnn.pad_packed_sequence(
                    rnn_out, batch_first=True)
            mean_rnn_out = torch.mean(rnn_out[0], dim=1)
            output = torch.cat([mean_rnn_out, output], dim=1)
        if self._encode_all:
            mean_x_embed = torch.mean(x_embed, dim=1)
            output = torch.cat([mean_x_embed, output], dim=1)
            output = self._linear(output)
        return output

class VideoCaption(BaseModel):
    def __init__(self, args, train=True):
        super(VideoCaption, self).__init__(args)
        self._video_rnn = VideoRNN(args, train=train)
        self._caption_rnn = CaptionRNN(args, train=train)

        if self._dist_fn == 'L2':
            self._dist_fn = F.pairwise_distance
        elif self._dist_fn == 'cossim':
            self._dist_fn = lambda x, y: -F.cosine_similarity(x, y)
        else:
            raise Exception('Unknown distance function')

        if train == False:
            self.load(self._load_model_filename)

    def forward(self, video, c_caption, c_length, w_caption, w_length):
        pred_video = self._video_rnn(video)
        pred_c = self._caption_rnn(c_caption, c_length)
        pred_w = self._caption_rnn(w_caption, w_length)
        return pred_video, pred_c, pred_w

    def count_triplet(self, video, c_caption, w_caption):
        c_distance = self._dist_fn(video, c_caption)
        w_distance = self._dist_fn(video, w_caption)
        return c_distance, w_distance

    def infer(self, video, caption, length):
        # infer testing data
        # video = [batch, 80, 4096]
        # caption = [batch*5, max_length]
        pred_video = self._video_rnn(video)
        pred_captions = self._caption_rnn(caption, length)
        return pred_video, pred_captions

    def count_distance(self, pred_video, pred_caption):
        # input arguments:
        # pred_video = [batch, hidden_size]
        # pred_caption = [batch*5, hidden_size]
        # output:
        # output = [batch, 5]
        batch_size = pred_video.shape[0]
        caption_len = pred_caption.shape[0]
        if caption_len % 5 != 0 or caption_len / 5 != batch_size:
            raise Exception('Predicted caption shape should be multiple of 5')
        pred_video = pred_video.repeat(1, 5).view(5*batch_size, -1)
        distances = self._dist_fn(pred_video, pred_caption)
        distances = distances.view(-1, 5)
        return distances

    def count_argmin_distance(self, pred_video, pred_caption):
        # TODO: simplize using count_distance
        # input arguments:
        # pred_video = [batch, hidden_size]
        # pred_caption = [batch*5, hidden_size]
        # output:
        # output = [batch]
        batch_size = pred_video.shape[0]
        caption_len = pred_caption.shape[0]
        if caption_len % 5 != 0 or caption_len / 5 != batch_size:
            raise Exception('Predicted caption shape should be multiple of 5')
        pred_video = pred_video.repeat(1, 5).view(5*batch_size, -1)
        distances = self._dist_fn(pred_video, pred_caption)
        distances = distances.view(-1, 5)
        output = torch.argmin(distances, dim=1)
        return output

    def save(self, filename):
        state_dict = {name:value.cpu() for name, value \
                in self.state_dict().items()}
        status = {'state_dict':state_dict,}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)
    
    def load(self, filename):
        status = torch.load(filename)
        self.load_state_dict(status['state_dict'])
    
class VCCommonSpace(VideoCaption):
    # Video Caption model with common space learning
    def __init__(self, args, train=True):
        super(VCCommonSpace, self).__init__(args, train=True)

        self._cs_in_space_dim = self._video_rnn._out_space_dim
        self._common_space = CommonSpace(
                self._cs_in_space_dim, self._common_space_dim)

        if train == False:
            self.load(self._load_model_filename)

    def forward(self, video, c_caption, c_length, w_caption, w_length):
        pred_video = self._video_rnn(video)
        pred_c = self._caption_rnn(c_caption, c_length)
        pred_w = self._caption_rnn(w_caption, w_length)

        # Common space
        pred_video = self._common_space(pred_video)
        pred_c = self._common_space(pred_c)
        pred_w = self._common_space(pred_w)
        return pred_video, pred_c, pred_w

    def infer(self, video, caption, length):
        # infer testing data
        # video = [batch, 80, 4096]
        # caption = [batch*5, max_length]
        pred_video = self._video_rnn(video)
        pred_captions = self._caption_rnn(caption, length)

        # Common space
        pred_video = self._common_space(pred_video)
        pred_captions = self._common_space(pred_captions)
        return pred_video, pred_captions

class CommonSpace(torch.nn.Module):
    def __init__(self, in_space_dim, out_space_dim=128):
        super(CommonSpace, self).__init__()
        self._init_network(in_space_dim, out_space_dim)

    def _init_network(self, in_space_dim, out_space_dim):
        self._linear = torch.nn.Sequential(
                    torch.nn.Linear(in_space_dim, out_space_dim),
                    torch.nn.BatchNorm1d(out_space_dim))

    def forward(self, x):
        return self._linear(x)
