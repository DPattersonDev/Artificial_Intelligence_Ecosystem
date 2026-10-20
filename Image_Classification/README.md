# AI Image Processing and Classification Project

## Project Image

For this project I used a personal photo of my cat, **Blueberry**.

Original image: `blueberry.jpg`

---

## Part 1: Image Classification and Grad-CAM

### AI Explanation of the Starter Classifier

The starter classifier uses TensorFlow and the pretrained MobileNetV2 model. The image is resized to 224 x 224 pixels, converted into a numerical array, preprocessed for MobileNetV2, and passed through the model. The program then decodes and prints the three ImageNet classes with the highest confidence scores.

The main parts of the program made sense to me because I have prior programming experience, although the TensorFlow-specific preprocessing and model calls were newer to me. AI assistance was useful for explaining why the image needs to be converted into an array and why a batch dimension is added before prediction.

### Top-3 Predictions

After running the classifier on `blueberry.jpg`, the model produced the following results:

1. **radiator** - **33.18%**
2. **sliding_door** - **25.27%**
3. **hognose_snake** - **2.23%**

The classifier did not identify Blueberry as a cat, which shows how strongly the surrounding environment can influence an image classification model.

### Grad-CAM

I used AI assistance to add Grad-CAM to the classifier. Grad-CAM works by examining the gradients flowing into the model's last convolutional layer for a selected prediction. Those gradients show which feature maps had the strongest influence on the prediction. The program combines those weighted feature maps into a heatmap and overlays it on the original image.

The resulting file is saved automatically as:

`blueberry_gradcam.jpg`

### Heatmap Observations

The Grad-CAM results suggest that the model focused more strongly on parts of the background and surrounding shapes than on Blueberry herself. This makes sense because the top predictions were "radiator" and "sliding_door," which are both related to objects or structures in the environment rather than the cat.

The result shows that image classifiers can sometimes be distracted by the setting of an image and may not correctly identify the main subject when the background contains strong visual features.

---

## Part 2: Image Filters

### Starter Blur Filter

The original `basic_filter.py` opens an image using Pillow, resizes it, and applies `ImageFilter.GaussianBlur(radius=2)`. The filtered image is then saved to a new file.

For Blueberry's image, the blur output is saved as:

`blueberry_blurred.jpg`

### Custom Storybook Filter

For my custom filter, I asked AI to help create a **storybook / illustrated effect**.

The filter:

- smooths small details,
- reduces the number of colors using posterization,
- increases color saturation,
- increases contrast,
- detects image edges, and
- blends the edge details back into the image to create an illustrated appearance.

The final output is saved as:

`blueberry_storybook.jpg`

### Filter Observations

The storybook filter keeps the subject recognizable while making the colors stronger and simplifying fine details. The edge processing adds a lightly drawn appearance, especially around Blueberry's face, clothing, and body.

Compared with the blur filter, the custom filter changes the style of the image rather than simply reducing detail.

---

## Final Reflection

This project helped me understand the difference between using a pretrained image classifier and directly processing image pixels with filters. MobileNetV2 analyzes patterns learned from ImageNet training data and produces confidence scores for classes that it believes are present in the image. Grad-CAM makes that process easier to interpret because it creates a visual representation of which regions had the greatest influence on the model's prediction.

In my test image, the classifier did not identify Blueberry as a cat. Instead, its highest predictions were "radiator" and "sliding_door." This helped me understand that an image classifier can sometimes focus more on the surrounding environment than on the subject a person expects it to recognize. The Grad-CAM heatmap made that behavior easier to examine because it showed which areas of the image contributed most strongly to the prediction.

Working with AI was especially useful when adding Grad-CAM because that required TensorFlow concepts that were less familiar to me, such as gradients, convolutional feature maps, and `GradientTape`. I still needed to understand how the code connected to the existing classifier rather than simply copying it. AI also helped explain the starter blur filter and design the storybook filter.

One challenge was keeping the changes understandable instead of adding unnecessary complexity. For the custom filter, I wanted an effect that was visibly different from the original but still easy to explain. Combining smoothing, posterization, color changes, and edge detection produced a noticeable result while using straightforward Pillow operations.

Overall, the project showed me how AI can help with both understanding unfamiliar code and iteratively developing new features while I still make decisions about the final design and evaluate the results.

---

## Files

- `blueberry.jpg` - original image
- `blueberry_gradcam.jpg` - Grad-CAM output
- `blueberry_blurred.jpg` - blur output
- `blueberry_storybook.jpg` - custom filter output
- `base_classifier.py` - MobileNetV2 classifier with Grad-CAM
- `basic_filter.py` - blur and storybook filters
- `requirements.txt` - required Python packages
