import os

file_path = "/root/Work/fuzzy-imager/venv310/lib/python3.10/site-packages/realesrgan/utils.py"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

with open(file_path, "r") as file:
    lines = file.readlines()

new_lines = []
patched = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Locate the model.load_state_dict block and patch it
    if 'model.load_state_dict(loadnet[keyname], strict=True)' in line:
        # Remove the original line
        new_lines.pop()
        patched = True

        patch_code = [
            "        # Handle both structured and raw checkpoints\n",
            "        if isinstance(loadnet, dict) and ('params_ema' in loadnet or 'params' in loadnet):\n",
            "            keyname = 'params_ema' if 'params_ema' in loadnet else 'params'\n",
            "            model.load_state_dict(loadnet[keyname], strict=True)\n",
            "        else:\n",
            "            model.load_state_dict(loadnet, strict=True)\n"
        ]
        new_lines.extend(patch_code)

if patched:
    with open(file_path, "w") as file:
        file.writelines(new_lines)
    print("✅ Patched realesrgan/utils.py successfully!")
else:
    print("⚠️ Patch not applied. Pattern not found.")

