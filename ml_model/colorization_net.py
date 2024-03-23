import torch.nn as nn


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128x128
            # You can add more layers here
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  # Maintains 256x256 output
            # Ensure output matches target size, adjust layers accordingly
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
