import torch


class SimpleCNN(torch.nn.Module):
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

    def forward(self, x):
        batch_size = x.shape[0]
        x = self._model_conv1(x)
        x = self._model_conv2(x)
        x = self._model_conv3(x)
        x = x.view(batch_size, -1)
        y = self._model_linear(x)
        prob = self._softmax(y)
        return prob

class Mobilenet(torch.nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self._init_network()

    def _init_network(self):
        def _conv_bn(in_channel, out_channel, stride):
            conv_bn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channel, out_channel, 3, stride,
                            1, bias=False),
                        torch.nn.BatchNorm2d(out_channel),
                        torch.nn.ReLU(inplace=True),
                    )
            return conv_bn
        def _conv_dw(in_channel, out_channel, stride):
            conv_dw = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channel, in_channel, 3, stride, 1,
                            groups=in_channel, bias=False),
                        torch.nn.BatchNorm2d(in_channel),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channel, out_channel, 1, 1, 0,
                            bias=False),
                        torch.nn.BatchNorm2d(out_channel),
                        torch.nn.ReLU(inplace=True),
                    )
            return conv_dw

        self._model_conv = torch.nn.Sequential(
                    _conv_bn(  1,  32, 2),
                    _conv_dw( 32,  64, 1),
                    _conv_dw( 64, 128, 2),
                    _conv_dw(128, 128, 1),
                    _conv_dw(128, 256, 2),
                    _conv_dw(256, 256, 1),
                    _conv_dw(256, 512, 2),
                    _conv_dw(512, 512, 1),
                    torch.nn.AvgPool2d(3),
                )
        self._model_linear = torch.nn.Sequential(
                    torch.nn.Linear(512, 64),
                    torch.nn.Linear(64, 7),
                )
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        conv = self._model_conv(x)
        linear_in = conv.view(batch_size, -1)
        linear_out = self._model_linear(linear_in)
        output = self._softmax(linear_out)
        return output
