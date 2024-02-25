import torch.nn as nn
import torch


class ConvBlock(nn.Module):

    def __init__(
            self,
            num_inp_channels: int,
            num_out_fmaps: int,
            kernel_size: int,
            pool_size: int = 2
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=num_inp_channels,
                              out_channels=num_out_fmaps,
                              kernel_size=(kernel_size, kernel_size),
                              bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool_2d = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_2d(self.relu(self.conv(x)))


class WildfireBinClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = ConvBlock(num_inp_channels=3, num_out_fmaps=8, kernel_size=2)
        self.conv2 = ConvBlock(num_inp_channels=8, num_out_fmaps=16, kernel_size=2)
        self.conv3 = ConvBlock(num_inp_channels=16, num_out_fmaps=32, kernel_size=2)

        # Defining the fully connected layers
        self.mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features=56448, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=300, out_features=2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        bsz, nch, height, width = x.shape
        x = torch.reshape(x, (bsz, (nch * height * width)))
        y = self.mlp(x)
        return y
