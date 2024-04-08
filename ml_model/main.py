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

file_name = 'colorization_model.pth'

# check to see if colorization_model.pth exists, if not, train the model
model = None
if not os.path.exists(file_name):
    print('Training model')
    model = train_model.train_model(file_name)
else:
    print('Model already trained')
    model = ColorizationNet()
    model.load_state_dict(torch.load(file_name))
# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
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

    # Colorize the image
    with torch.no_grad():
        output = model(input_image)

    # Convert the output tensor to an image
    output_image = output.squeeze().cpu().detach()
    output_image = to_pil_image(output_image)

    # Save the image to a BytesIO object
    image_bytes = BytesIO()
    output_image.save(image_bytes, format='JPEG')
    img_str = base64.b64encode(image_bytes.getvalue()).decode()

    return {"image": img_str}


@app.on_event("startup")
async def startup():
    # no static database connection for now, until I consider limiting the number of usages per IP
    pass


@app.on_event("shutdown")
async def shutdown():
    pass
