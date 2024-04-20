# https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from image_colorization_dataset import ImageColorizationDataset
from colorization_net import ColorizationNet


def psnr(label, output, max_pixel=1.0):
    mse = torch.mean((label - output) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def train_model(file_name):
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),

    ])

    dataset = ImageColorizationDataset('data/train_black', 'data/train_color', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)


    print('dataset loaded successfully!')
    # use ColorizationNet to train the model 5 epoch
    model = ColorizationNet().to(device)

    # custom loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['black_image'].to(device), data['color_image'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # use custom loss function
            crit = criterion(outputs, labels)

            loss = crit
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch: ', epoch, 'Loss: ', running_loss / len(dataloader))


    print('Finished Training')

    # load model colorization_model.pth
    # model.load_state_dict(torch.load('colorization_model.pth'))

    # evaluate the model on the test set
    test_dataset = ImageColorizationDataset('data/test_black', 'data/test_color', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Test using MSE loss
    test_loss = 0.0
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data['black_image'].to(device), data['color_image'].to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
    print('Average test loss: ', test_loss / len(test_dataloader))

    # save the model
    torch.save(model.state_dict(), file_name)
    return model


def rgb_to_hsv(image):
    # converts an RGB image to an HSV image
    # this makes it easier to extract the saturation channel
    # as well as manage the loss function
    max_channel, _ = image.max(dim=1, keepdim=True)
    min_channel, _ = image.min(dim=1, keepdim=True)
    channel_difference = max_channel - min_channel  # difference between the max and min channel,
    # aka value channel, aka the difference between the brightest and darkest channel
    value = max_channel  # value channel, which is the max channel
    saturation = channel_difference / (max_channel + 1e-6)  # saturation channel, aka how colorful the pixel is

    # calculate the hue channel
    r, g, b = image.split(1, dim=1)
    hue = torch.zeros_like(r)

    # Red is max
    idx = (max_channel == r)
    hue[idx] = (g[idx] - b[idx]) / (channel_difference[idx] + 1e-6)

    # Green is max
    idx = (max_channel == g)
    hue[idx] = 2.0 + (b[idx] - r[idx]) / (channel_difference[idx] + 1e-6)

    # Blue is max
    idx = (max_channel == b)
    hue[idx] = 4.0 + (r[idx] - g[idx]) / (channel_difference[idx] + 1e-6)

    hue = (hue / 6.0) % 1.0  # normalize the hue channel to be between 0 and 1
    return torch.cat([hue, saturation, value], dim=1)


def l2_saturation_loss(output, target):
    # modified loss function that penalizes the model for producing images with low saturation

    output_hsv = rgb_to_hsv(output)  # convert the output image to HSV, so we can extract the saturation channel
    target_hsv = rgb_to_hsv(target)  # convert the target image to HSV, so we can extract the saturation channel

    # extract the saturation channels by getting the second channel (saturation) from the HSV images (index 1)
    output_sat = output_hsv[:, 1, :, :]
    target_sat = target_hsv[:, 1, :, :]

    # calculate the mean saturation value for the target image
    mean_target_sat = target_sat.mean()

    # calculate the mean saturation value for the output image
    mean_output_sat = output_sat.mean()

    # calculate the mean saturation difference
    mean_diff = torch.abs(mean_output_sat - mean_target_sat)

    # Calculate the return loss
    return mean_diff
