import os
from PIL import Image

def resize_image(image, target_size=(256, 256)):
    """等比压缩或拉伸图像到目标大小"""
    return image.resize(target_size, Image.BICUBIC)  # 也可用 LANCZOS 或 ANTIALIAS

def batch_resize_images(input_dir, output_dir, target_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            try:
                with Image.open(input_path) as img:
                    resized = resize_image(img, target_size)
                    resized.save(output_path)
                    print(f'Resized and saved: {fname}')
            except Exception as e:
                print(f'Skipped {fname} due to error: {e}')

# 用法
input_folder = '../Hubble_Images_top90'
output_folder = '../Hubble_Images_top90_256x256'
batch_resize_images(input_folder, output_folder)
