import argparse
from enhancer.processor import enhance_image

parser = argparse.ArgumentParser(description="AI Image Enhancer")
parser.add_argument("--input", required=True, help="Path to input image")
parser.add_argument("--output", required=True, help="Path to save enhanced image")
parser.add_argument("--scale", type=int, choices=[2, 4, 8], default=2, help="Upscale factor")

args = parser.parse_args()

enhance_image(args.input, args.output, scale=args.scale)
print("✅ Image enhancement complete!")
