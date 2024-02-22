import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


def preprocess_single_image(img_path, args):
    out_dir = os.path.dirname(img_path)
    out_rgba = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_rgba.png')
    out_caption = os.path.join(out_dir, os.path.basename(img_path).split('.')[0] + '_caption.txt')

    # load image
    print(f'[INFO] loading image {img_path}...')

    # check the exisiting files
    if os.path.isfile(out_rgba): 
        print(f"{img_path} has already been here!")
        return
    print(img_path)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    carved_image = None

    if image.shape[-1] == 4:
        carved_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if carved_image is None:
        # carve background
        print(f'[INFO] background removal...')
        carved_image = BackgroundRemoval()(image) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    if opt.recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(opt.size * (1 - opt.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (opt.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (opt.size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    else:
        final_rgba = carved_image

    # write output
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))

    if opt.do_caption:
        # predict caption (it's too slow... use your brain instead)
        print(f'[INFO] captioning...')
        blip2 = BLIP2()
        caption = blip2(image)
        with open(out_caption, 'w') as f:
            f.write(caption)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--size', default=1024, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.1, type=float, help="output border ratio")
    parser.add_argument('--recenter', action='store_true', help="recenter, potentially not helpful for multiview zero123")
    parser.add_argument('--do_caption', action='store_true', help="do text captioning")
    
    opt = parser.parse_args()

    if os.path.isdir(opt.path):
        img_list = sorted(os.path.join(root, fname) for root, _dirs, files in os.walk(opt.path) for fname in files)
        img_list = [img for img in img_list if not img.endswith("rgba.png")] 
        img_list = [img for img in img_list if img.endswith(".png")]
        for img in img_list:
            preprocess_single_image(img, opt)
    else: # single image file
        preprocess_single_image(opt.path, opt)