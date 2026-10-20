import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os

# Load the pretrained MobileNetV2 classifier.
model = MobileNetV2(weights="imagenet")

# MobileNetV2's last convolutional layer is used for Grad-CAM.
LAST_CONV_LAYER = "Conv_1"


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Create a Grad-CAM heatmap for the model's predicted class."""

    # Build a model that returns both convolutional features and predictions.
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record calculations so TensorFlow can compute gradients.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Use the highest-scoring prediction unless a class is provided.
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Measure how strongly each feature map affected the selected class.
    grads = tape.gradient(class_channel, conv_outputs)

    # Average gradients across the spatial dimensions.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by their importance.
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Keep positive contributions and normalize to 0-1.
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def save_gradcam_overlay(image_path, heatmap, output_path, alpha=0.4):
    """Overlay the Grad-CAM heatmap on the original image."""

    original = cv2.imread(image_path)

    if original is None:
        raise ValueError(f"Could not open image: {image_path}")

    resized_heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    resized_heatmap = np.uint8(255 * resized_heatmap)
    colored_heatmap = cv2.applyColorMap(
        resized_heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )

    cv2.imwrite(output_path, overlay)


def classify_image(image_path):
    """Classify an image, print the top three results, and save Grad-CAM."""

    try:
        # Resize to the size MobileNetV2 expects.
        img = image.load_img(image_path, target_size=(224, 224))

        # Convert image pixels to a numerical array.
        img_array = image.img_to_array(img)

        # Apply MobileNetV2 preprocessing.
        img_array = preprocess_input(img_array)

        # Add a batch dimension.
        img_array = np.expand_dims(img_array, axis=0)

        # Run the classifier.
        predictions = model.predict(img_array)

        # Decode the top three ImageNet predictions.
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2%})")

        # Generate the Grad-CAM heatmap.
        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            LAST_CONV_LAYER
        )

        base_name, _ = os.path.splitext(image_path)
        gradcam_path = f"{base_name}_gradcam.jpg"

        save_gradcam_overlay(
            image_path,
            heatmap,
            gradcam_path
        )

        print(f"Grad-CAM image saved as '{gradcam_path}'.")

    except Exception as e:
        print(f"Error processing '{image_path}': {e}")


if __name__ == "__main__":
    print("Image Classifier with Grad-CAM (type 'exit' to quit)\n")

    while True:
        image_path = input("Enter image filename: ").strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        classify_image(image_path)
