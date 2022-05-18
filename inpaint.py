import os
from omegaconf import OmegaConf
import numpy as np
import torch
import cv2
import albumentations
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
# pip install pytorch-lightning==1.6.1

STEPS = 50
IMG_PATH = 'data/09.jpg'
OUTPUT_PATH = 'outputs'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load a model
# Pretrained model https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
config = OmegaConf.load('models/ldm/inpainting_big/config.yaml')
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load('models/ldm/inpainting_big/last.ckpt')['state_dict'], strict=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
sampler = DDIMSampler(model)

img = cv2.imread(IMG_PATH)

# Center crop
if img.shape[0] < img.shape[1]:
    img = cv2.resize(img, (int(512 / img.shape[0] * img.shape[1]), 512))
else:
    img = cv2.resize(img, (512, int(512 / img.shape[1] * img.shape[0])))
img = albumentations.CenterCrop(height=512, width=512)(image=img)['image']

img_ori = img.copy()
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)

# Draw mask
drawing = False
def draw_circle(event, x, y, flags, param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 20, (0, 0, 0), -1)
            cv2.circle(mask, (x, y), 20, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 20, (0, 0, 0), -1)
        cv2.circle(mask, (x, y), 20, (0, 0, 0), -1)

cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)

while True:
    cv2.imshow('img', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        img = img_ori.copy()
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    elif key == ord('w'): # Inpaint
        masked_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img = masked_img.astype(np.float32) / 255.
        masked_img = np.expand_dims(masked_img, axis=0)
        masked_img = np.transpose(masked_img, (0, 3, 1, 2))
        masked_img = torch.from_numpy(masked_img)
        masked_img = masked_img.to(device)
        masked_img = masked_img * 2. - 1.

        mask_ori = np.expand_dims(mask.copy(), axis=-1)
        mask_input = np.expand_dims(mask, axis=(0, 1))
        mask_input = torch.from_numpy(mask_input)
        mask_input = mask_input.to(device)
        mask_input = (1. - mask_input) * 2. - 1.

        with torch.no_grad():
            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(masked_img)
            cc = torch.nn.functional.interpolate(mask_input, size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            samples_ddim, _ = sampler.sample(
                S=STEPS,
                conditioning=c,
                batch_size=c.shape[0],
                shape=(c.shape[1] - 1,) + c.shape[2:],
                verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            predicted_img = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_img = predicted_img.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
            predicted_img = cv2.cvtColor(predicted_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

            inpainted_img = mask_ori * img_ori + (1 - mask_ori) * predicted_img
            inpainted_img = inpainted_img.astype(np.uint8)

            cv2.imshow('output', inpainted_img)
