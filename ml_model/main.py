# https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from ImageColorizationDataset import ImageColorizationDataset


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


def psnr(label, output, max_pixel=1.0):
    mse = torch.mean((label - output) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


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

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['black_image'], data['color_image']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1250 == 1249:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

    print('Finished Training')

    # load model colorization_model.pth
    # model.load_state_dict(torch.load('colorization_model.pth'))

    # evaluate the model on the test set
    test_dataset = ImageColorizationDataset('data/test_black', 'data/test_color', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    correct = 0
    total = 0

    # Example usage in your evaluation loop
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data['black_image'], data['color_image']
            outputs = model(inputs)
            psnr_value = psnr(labels, outputs)
            print(f'PSNR value: {psnr_value.item()} dB')

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

    # try to use model to colorize a black and white image
    # open myphoto.jpg
    # transform myphoto.jpg
    # use model to colorize the image
    # save the colorized image as myphoto_colorized.jpg

    model = ColorizationNet()
    model.load_state_dict(torch.load('colorization_model.pth'))
    model.eval()

    image_path = 'myphoto.jpg'
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Colorize the image
    with torch.no_grad():
        output = model(input_image)

    # Convert the output tensor to an image
    output_image = output.squeeze().cpu().detach()
    output_image = to_pil_image(output_image)
    # denormalize the image


    output_image.save('myphoto_colorized.jpg')

    # image colorization dataset, downloaded from kaggle
    # in the /data folder organized as follows:
    # data/ - holds the dataset
    # data/test_black/ - holds the black and white images for validation
    # data/test_color/ - holds the colored images for validation
    # data/train_black/ - holds the black and white images for training
    # data/train_color/ - holds the colored images for training


if __name__ == '__main__':
    main()
