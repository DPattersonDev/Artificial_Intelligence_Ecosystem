from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import os


def apply_blur_filter(image_path, output_path="blurred_image.png"):
    """Apply the starter Gaussian blur filter."""

    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        img_blurred = img_resized.filter(
            ImageFilter.GaussianBlur(radius=2)
        )

        plt.imshow(img_blurred)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Blurred image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_storybook_filter(image_path, output_path="storybook_image.jpg"):
    """Create a colorful illustrated/storybook effect."""

    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((512, 512))

        # Smooth small details.
        smoothed = img_resized.filter(ImageFilter.SMOOTH_MORE)

        # Reduce the number of colors for a painted look.
        posterized = ImageOps.posterize(smoothed, 4)

        # Increase saturation and contrast.
        colorful = ImageEnhance.Color(posterized).enhance(1.45)
        colorful = ImageEnhance.Contrast(colorful).enhance(1.20)

        # Create dark edge details.
        edges = img_resized.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges = ImageEnhance.Contrast(edges).enhance(2.0)
        edge_rgb = Image.merge("RGB", (edges, edges, edges))

        # Blend color and edge details together.
        storybook = Image.blend(colorful, edge_rgb, 0.18)

        storybook.save(output_path)
        print(f"Storybook image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Image Processor (type 'exit' to quit)")
    print("Available filters: blur, storybook\n")

    while True:
        image_path = input(
            "Enter image filename (or 'exit' to quit): "
        ).strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        choice = input(
            "Choose filter ('blur' or 'storybook'): "
        ).strip().lower()

        base_name, ext = os.path.splitext(image_path)

        if choice == "blur":
            output_file = f"{base_name}_blurred{ext}"
            apply_blur_filter(image_path, output_file)

        elif choice == "storybook":
            output_file = f"{base_name}_storybook.jpg"
            apply_storybook_filter(image_path, output_file)

        else:
            print("Unknown filter choice. Please select 'blur' or 'storybook'.")
