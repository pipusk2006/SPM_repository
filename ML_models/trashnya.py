import os
from PIL import Image

def scan_cache_for_images(cache_dir):
    for root, _, files in os.walk(cache_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img.show()  # Показать найденное изображение
                except:
                    continue

# Путь к кешу Telegram (пример для Windows)
telegram_cache = r"C:\Users\<USER>\AppData\Roaming\Telegram Desktop\tdata"
scan_cache_for_images(telegram_cache)