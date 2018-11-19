import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
    
    def forward(self, x):
        batch_size = x.shape[0]
        conv = self._model_conv(x)
        linear_in = conv.view(batch_size, -1)
        linear_out = self._model_linear(linear_in)
        output = self._softmax(linear_out)
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

class SimpleCNN(CNN):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self._init_network()

    def _init_network(self):
        self._model_conv1 = torch.nn.Sequential(
                    # input_channel, out_channel, filter_size, stride, padding
                    torch.nn.Conv2d(1, 16, 5, 1, 2),
                    torch.nn.MaxPool2d(2),
                    torch.nn.ReLU(),
                )
        self._model_conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(16, 32, 5, 1, 2),
                    torch.nn.Dropout2d(p=0.2),
                    torch.nn.MaxPool2d(2),
                    torch.nn.ReLU(),
                )
        self._model_conv3 = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 5, 1, 2),
                    torch.nn.MaxPool2d(2),
                    torch.nn.ReLU(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(6*6*64, 256),
                    torch.nn.Linear(256, 64),
                    torch.nn.Linear(64, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)
        self._model_conv = torch.nn.Sequential(
                    self._model_conv1,
                    self._model_conv2,
                    self._model_conv3,
                )

class Mobilenet(CNN):
    # Inspired from mobilnet 
    # use the depthwise convolution
    def __init__(self):
        super(Mobilenet, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.ReLU(inplace=True)
        def _dropout():
            return torch.nn.Dropout()
        def _conv_bn(in_channel, out_channel, stride):
            conv_bn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channel, out_channel, 3, stride,
                            1, bias=False),
                        torch.nn.BatchNorm2d(out_channel),
                        _relu(),
                    )
            return conv_bn
        def _conv_dw(in_channel, out_channel, stride):
            conv_dw = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channel, in_channel, 3, stride, 1,
                            groups=in_channel, bias=False),
                        torch.nn.BatchNorm2d(in_channel),
                        _relu(),
                        torch.nn.Conv2d(in_channel, out_channel, 1, 1, 0,
                            bias=False),
                        torch.nn.BatchNorm2d(out_channel),
                        _relu(),
                    )
            return conv_dw

        self._model_conv = torch.nn.Sequential(
                    _conv_bn(  1,  16, 1),
                    _conv_bn( 16,  32, 1),
                    _conv_bn( 32,  32, 2),
                    _conv_dw( 32,  64, 1),
                    _conv_dw( 64, 128, 2),
                    _conv_dw(128, 256, 2),
                    _conv_dw(256, 512, 2),
                    #torch.nn.AvgPool2d(2),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    torch.nn.BatchNorm1d(512),
                    _dropout(),
                    _relu(),
                    torch.nn.Linear(512, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)


class VGG(CNN):
    # Inspired from VGG network
    # Not obtaining great performance on this task
    def __init__(self):
        super(VGG, self).__init__()
        self._init_network()

    def _init_network(self):
        def _conv(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    torch.nn.ReLU(inplace=True),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout2d():
            return torch.nn.Dropout2d()
        def _dropout():
            return torch.nn.Dropout()
        self._model_conv = torch.nn.Sequential(
                    _conv(1, 16),
                    _conv(16, 32),
                    _max_pool(),
                    _conv(32, 64),
                    _conv(64, 64),
                    _max_pool(),
                    _dropout2d(),
                    _conv(64, 128),
                    _conv(128, 128),
                    _max_pool(),
                    _conv(128, 256),
                    _conv(256, 256),
                    _max_pool(),
                    _dropout2d(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(256*3*3, 256),
                    _dropout(),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(256, 32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(32, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

class TsungHan(CNN):
    # Model suggested from TsungHan
    # This model is used to compare with the other model performance
    # Reach about 66% for validation
    def __init__(self):
        super(TsungHan, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.ReLU(inplace=True)
        def _conv(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout():
            return torch.nn.Dropout()
        self._model_conv = torch.nn.Sequential(
                    _conv(1, 16),
                    _conv(16, 32),
                    _conv(32, 64),
                    _max_pool(),
                    _conv(64, 128),
                    _max_pool(),
                    _conv(128, 256),
                    _max_pool(),
                    _conv(256, 512),
                    _max_pool(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    torch.nn.BatchNorm1d(512),
                    _relu(),
                    _dropout(),
                    torch.nn.Linear(512, 256),
                    _relu(),
                    torch.nn.Linear(256, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

class Kai(CNN):
    # Inspired from TsungHan model
    # Kai takes more care about the later stage of convolution
    # and pays less attention to the early stage.
    def __init__(self):
        super(Kai, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.LeakyReLU(inplace=True)
        def _conv(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout():
            return torch.nn.Dropout()
        def _dropout2d():
            return torch.nn.Dropout2d()
        self._model_conv = torch.nn.Sequential(
                    _conv(1, 16),
                    _conv(16, 32),
                    _max_pool(),
                    _conv(32, 64),
                    _conv(64, 128),
                    _max_pool(),
                    _conv(128, 256),
                    _conv(256, 256),
                    _max_pool(),
                    _dropout2d(),
                    _conv(256, 512),
                    _conv(512, 512),
                    _max_pool(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    #torch.nn.BatchNorm1d(512),
                    _relu(),
                    _dropout(),
                    torch.nn.Linear(512, 256),
                    _relu(),
                    torch.nn.Linear(256, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

class Kai2(CNN):
    # which is a trying from Kai that the convolution kernel
    # size alters.
    def __init__(self):
        super(Kai2, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.ReLU(inplace=True)
        def _conv5(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 5, 1, 2),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _conv3(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout():
            return torch.nn.Dropout()
        self._model_conv = torch.nn.Sequential(
                    _conv5(1, 16),
                    _conv5(16, 32),
                    _max_pool(),
                    _conv5(32, 64),
                    _conv5(64, 128),
                    _max_pool(),
                    _conv3(128, 256),
                    _conv3(256, 256),
                    _max_pool(),
                    _conv3(256, 512),
                    _conv3(512, 512),
                    _max_pool(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    torch.nn.BatchNorm1d(512),
                    _relu(),
                    _dropout(),
                    torch.nn.Linear(512, 256),
                    _relu(),
                    torch.nn.Linear(256, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

class Kai3(CNN):
    # A transform from Kai
    # It pays more attention on the early stage of convolution
    def __init__(self):
        super(Kai3, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.ReLU(inplace=True)
        def _conv(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout():
            return torch.nn.Dropout()
        def _dropout2d():
            return torch.nn.Dropout2d()
        self._model_conv = torch.nn.Sequential(
                    _conv(1, 16),
                    _conv(16, 16),
                    _conv(16, 32),
                    _max_pool(),
                    _conv(32, 64),
                    _conv(64, 64),
                    _conv(64, 128),
                    _max_pool(),
                    _dropout2d(),
                    _conv(128, 256),
                    _max_pool(),
                    _conv(256, 512),
                    _max_pool(),
                    _dropout2d(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    torch.nn.BatchNorm1d(512),
                    _relu(),
                    _dropout(),
                    torch.nn.Linear(512, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

class Kai4(CNN):
    def __init__(self):
        super(Kai4, self).__init__()
        self._init_network()

    def _init_network(self):
        def _relu():
            return torch.nn.ReLU(inplace=True)
        def _conv(in_channel, out_channel):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                    torch.nn.BatchNorm2d(out_channel),
                    _relu(),
                )
            return conv
        def _max_pool():
            return torch.nn.MaxPool2d(2)
        def _dropout():
            return torch.nn.Dropout()
        self._model_conv = torch.nn.Sequential(
                    _conv(1, 16),
                    _conv(16, 32),
                    _max_pool(),
                    _conv(32, 64),
                    _conv(64, 128),
                    _max_pool(),
                    _conv(128, 256),
                    _max_pool(),
                    _conv(256, 512),
                    _max_pool(),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512*3*3, 512),
                    torch.nn.BatchNorm1d(512),
                    _relu(),
                    _dropout(),
                    torch.nn.Linear(512, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)
