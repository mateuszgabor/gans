import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def add_epoch_title(image, epoch, font_size):
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        print("DejaVuSans-Bold.ttf not found. Using default font.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    text = f"Epoch: {epoch}"
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

    new_image = Image.new(
        "RGB", (image.width, image.height + text_height + 20), "black"
    )
    draw = ImageDraw.Draw(new_image)

    text_position = ((new_image.width - text_width) // 2, 10)
    draw.text(text_position, text, font=font, fill="white")
    new_image.paste(image, (0, text_height + 20))

    return new_image


def create_gif(image_folder, duration, loop, font_size):
    image_folder = Path(image_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"The image folder does not exist: {image_folder}")

    image_files = sorted(image_folder.glob("*.png"), key=lambda x: int(x.stem))

    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {image_folder}")

    images = []
    gif_path = f"{image_folder.parent.parent}/gifs"
    Path(gif_path).mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        img = Image.open(img_path)
        epoch = img_path.stem
        img = add_epoch_title(img, epoch, font_size)
        images.append(img)

    images[0].save(
        f"{gif_path}/{image_folder.name}.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True)
    parser.add_argument("-d", type=int, default=500, required=False)
    parser.add_argument("-l", type=int, default=0, required=False)
    parser.add_argument("-fs", type=int, default=36, required=False)

    args = parser.parse_args()

    create_gif(args.path, args.d, args.l, args.fs)
