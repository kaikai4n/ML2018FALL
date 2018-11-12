import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self._init_network()

    def _init_network(self):
        self.conv1 = torch.nn.Sequential(
                    # input_channel, out_channel, filter_size, stride, padding
                    torch.nn.Conv2d(1, 16, 5, 1, 2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                )
        self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(16, 32, 5, 1, 2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                )
        self.conv3 = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 5, 1, 2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                )
        self.linear = torch.nn.Sequential(
                    torch.nn.Linear(6*6*64, 1024),
                    torch.nn.Linear(1024, 512),
                    torch.nn.Linear(512, 7),
                )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        y = self.linear(x)
        prob = self.softmax(y)
        return prob
