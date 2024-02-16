# https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ImageColorizationDataset import ImageColorizationDataset


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as needed
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted
            # Add more layers as needed
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),

    ])

    dataset = ImageColorizationDataset('data/train_black', 'data/train_color', transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print('dataset loaded successfully!')
    # use ColorizationNet to train the model 5 epoch
    model = ColorizationNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['black_image'], data['color_image']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

    print('Finished Training')

    # evaluate the model on the test set
    test_dataset = ImageColorizationDataset('data/test_black', 'data/test_color', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    correct = 0
    total = 0

    # these are images so we need to compute the accuracy using MSE
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data['black_image'], data['color_image']
            outputs = model(inputs)
            total += (labels.size(0) * 256 * 256)
            correct += (torch.sum((outputs - labels) ** 2)).item()



    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    # save the model
    torch.save(model.state_dict(), 'colorization_model.pth')


    # image colorization dataset, downloaded from kaggle
    # in the /data folder organized as follows:
    # data/ - holds the dataset
    # data/test_black/ - holds the black and white images for validation
    # data/test_color/ - holds the colored images for validation
    # data/train_black/ - holds the black and white images for training
    # data/train_color/ - holds the colored images for training


if __name__ == '__main__':
    main()