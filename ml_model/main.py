import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

import train_model
from colorization_net import ColorizationNet
from PIL import Image
from io import BytesIO
import base64

file_names = []
for i in range(0, 125):
    file_name = 'colorization_model_combo_' + str(i) + '.pth'
    if os.path.exists(file_name):
        file_names.append(file_name)


# check to see if colorization_model.pth exists, if not, train the model
models = []
for file_name in file_names:
    if not os.path.exists(file_name):
        continue
    else:
        model = ColorizationNet()
        model.load_state_dict(torch.load(file_name))
        models.append(model)
# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for model in models:
    model.to(device)

for model in models:
    model.eval()

app = FastAPI()

# small app, shouldn't need to split into multiple files

origins_env = os.getenv("CORS_ORIGINS", "http://localhost:4200,https://localhost:4200,http://localhost")
origins = origins_env.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/colorize")
async def colorize_image(file: UploadFile = File(...)):
    # Verify that the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"error": "Invalid file type"}

    # Turn uploaded file into a PIL image
    image = Image.open(file.file).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    input_image = input_image.to(device)

    return_obj = {}

    for m in models:
        # Colorize the image using all models
        with torch.no_grad():
            output = m(input_image)

        # Convert the output tensor to an image
        output_image = output.squeeze().cpu().detach()
        output_image = to_pil_image(output_image)

        # Save the image to a BytesIO object
        image_bytes = BytesIO()
        output_image.save(image_bytes, format='JPEG')
        img_str = base64.b64encode(image_bytes.getvalue()).decode()

        return_obj[file_names[models.index(m)]] = img_str

    return return_obj


@app.on_event("startup")
async def startup():
    # no static database connection for now, until I consider limiting the number of usages per IP
    pass


@app.on_event("shutdown")
async def shutdown():
    pass
