# 🧠 Fuzzy Imager

Fuzzy Imager is a lightweight AI-powered image enhancement microservice built with **FastAPI**, **OpenCV**, and **Real-ESRGAN**. It supports classical image processing techniques and deep learning upscaling for high-quality image restoration.

---

## ✨ Features

- ✅ **Image Enhancement**: contrast improvement, color saturation, and sharpening.
- 🔍 **Deep Upscaling**: using Real-ESRGAN models (x2, x4, x8).
- 🚀 **Microservice API**: FastAPI-based endpoint, easily integratable.
- ⚙️ **CUDA/CPU Support**: works with or without a GPU.
- 🔧 **Patched Real-ESRGAN** for raw model support and better compatibility.
- 🧪 **cURL-friendly** usage and CLI integration.

---

## 📦 Project Structure

```bash
fuzzy-imager/
├── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── api.py           # HTTP route for /enhance
│   └── enhancer/
│       ├── processor.py # Contrast, color, sharpness
│       └── upscaler.py  # Deep learning (Real-ESRGAN)
├── weights/             # Real-ESRGAN .pth models (x2, x4, x8)
├── test_images/         # Example input and output
├── Real-ESRGAN/         # Submodule or manually integrated code
├── run.py               # Uvicorn launcher
└── requirements.txt
```

---

## 🚧 Real-ESRGAN Patch

### ❗ Issue

When using `.pth` weights directly with the `RealESRGANer`, an error is thrown:

```
KeyError: 'params'
```

This is because Real-ESRGAN expects `params_ema` or `params` in the checkpoint dictionary, but some `.pth` files are just raw model state dicts.

### ✅ Fix (Patch)

In `venv/.../realesrgan/utils.py`, modify this part in the `RealESRGANer.__init__` method:

```python
# PATCH: Handle both structured and raw checkpoints
if isinstance(loadnet, dict) and ('params_ema' in loadnet or 'params' in loadnet):
    keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
else:
    model.load_state_dict(loadnet, strict=True)  # <--- added fallback
```

This enables support for raw .pth files like those in weights/.

---

## 🧪 Example Usage

### 1. Start the API

```bash
python run.py
```

### 2. Send a request using curl

```bash
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@test_images/input.jpg" \
  -F "scale=4" \
  -L \
  --output test_images/enhanced_api.jpg
```

✅ The enhanced image will be saved as `test_images/enhanced_api.jpg`.

---

## 🧠 Setup

### Clone the repo

```bash
git clone https://github.com/alex22441/fuzzy-imager.git
cd fuzzy-imager
```

If you’re using Real-ESRGAN as a submodule, run:

```bash
git submodule update --init --recursive
```

### Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

🔄 Make sure to install numpy<2.0 if you're using PyTorch 1.13:

```bash
pip install "numpy<2.0"
```

---

## 🧩 Supported Models

Store your `.pth` model files in the `weights/` directory. The following are supported:

- RealESRGAN_x2.pth
- RealESRGAN_x4.pth
- RealESRGAN_x8.pth

More models available from: https://github.com/xinntao/Real-ESRGAN

---

## 🔥 Roadmap

- [ ] Docker image  
- [ ] Web UI for drag-and-drop enhancement  
- [ ] Multiple image batch support  
- [ ] Model benchmarking API  

---

## 👨‍💻 Maintainer

**Alex** [@alex22441](https://github.com/alex22441)  
🧠 HomeLab | DevOps | Discord Bots | AI SaaS | FastAPI Microservices  
