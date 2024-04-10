# https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    criterion_weights = [0, 1, 2, 3, 4] # testing loss function weights
    l2_saturation_loss_weights = [0, 1, 2, 3, 4] # testing loss function weights
    psnr_weights = [0, 1, 2, 3, 4] # testing loss function weights
    combo_num = 0
    total_ran = 0
    # unique triplets of weights
    unique_permutations = []


    for crit_w in criterion_weights:
        for l2_sat_loss_w in l2_saturation_loss_weights:
            for psnr_w in psnr_weights:
                file_name = 'colorization_model_combo_' + str(combo_num) + '.pth'
                combo_num += 1
                gcf = get_gcf(crit_w, l2_sat_loss_w, psnr_w)
                if gcf == 0:
                    continue
                p = (crit_w // gcf, l2_sat_loss_w // gcf, psnr_w // gcf)
                if p in unique_permutations:
                    print("Skipping, permutation already ran")
                    continue
                unique_permutations.append(p)

                total_ran += 1
                print("Training model number: %d with weights: %f, %f, %f ,Total ran: %d" % (combo_num, crit_w, l2_sat_loss_w, psnr_w, total_ran))

                model = ColorizationNet().to(device)

                for epoch in range(5):
                    running_loss = 0.0
                    running_psnr = 0.0
                    running_l2_saturation_loss = 0.0
                    running_criterion_loss = 0.0
                    for i, data in enumerate(dataloader, 0):
                        inputs, labels = data['black_image'].to(device), data['color_image'].to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        # use custom loss function
                        crit = criterion(outputs, labels)
                        l2_sat_loss = l2_saturation_loss(outputs, labels)
                        p = psnr(labels, outputs)

                        running_psnr += p.item()
                        running_l2_saturation_loss += l2_sat_loss.item()
                        running_criterion_loss += crit.item()

                        abs_root_l2_sat_loss = torch.sqrt(l2_sat_loss)

                        loss = crit * crit_w + (0.0001 * abs_root_l2_sat_loss) * l2_sat_loss_w + p * psnr_w
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % 1250 == 1249:
                            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                            print(f'[{epoch + 1}, {i + 1}] PSNR: {running_psnr / 2000}')
                            print(f'[{epoch + 1}, {i + 1}] L2 Saturation Loss: {running_l2_saturation_loss / 2000}')
                            print(f'[{epoch + 1}, {i + 1}] Criterion Loss: {running_criterion_loss / 2000}')
                            running_loss = 0.0
                            running_psnr = 0.0
                            running_l2_saturation_loss = 0.0
                            running_criterion_loss = 0.0
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


def get_gcf(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(a, b), c)
