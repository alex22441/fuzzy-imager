import os
import torch
import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_image(cv2_img, scale=2):
    # Validate scale
    if scale not in [2, 4, 8]:
        raise ValueError("❌ Invalid scale. Supported values are 2, 4, or 8.")

    # Path to model
    model_path = os.path.join("weights", f"RealESRGAN_x{scale}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    # Convert image
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img_np = np.array(img_pil)

    # Enhance image
    try:
        output, _ = upsampler.enhance(img_np, outscale=1)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    except RuntimeError as e:
        print("❌ Error during enhancement:", e)
        return cv2_img  # fallback
