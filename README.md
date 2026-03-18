#   Brain Tumor Classification

A deep learning web application that classifies brain MRI images to detect the presence of tumors using a Convolutional Neural Network (CNN), deployed with a Gradio interface.

---

##  Project Overview

This project uses a trained CNN model to analyze brain MRI scans and predict whether a tumor is present or not. Users can upload or select an image through a simple web UI powered by Gradio.

- **Input:** Brain MRI image
- **Output:** `"It's a Tumor"` or `"No, It's not a Tumor"` with confidence score
- **Interface:** Gradio web app (shareable public link)

---

##  Dataset

- **Location:** `brain_tumor_dataset/` with two subdirectories:
  - `yes/` — MRI images with tumors
  - `no/` — MRI images without tumors
- **Total images:** 139
- **Train/Test split:** 111 training / 28 testing (80/20, `random_state=0`)
- **Image size:** Resized to `128 × 128 × 3` (RGB)

---

##  Model Architecture

```
Sequential CNN
├── Conv2D(32, 2×2) → BatchNormalization → MaxPooling2D → Dropout(0.25)
├── Conv2D(64, 2×2) → BatchNormalization → MaxPooling2D → Dropout(0.25)
├── Flatten
├── Dense(512, activation='relu') → Dropout(0.5)
└── Dense(2, activation='softmax')
```

- **Total Parameters:** 33,585,602 (~128 MB)
- **Optimizer:** Adamax
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 25 | **Batch Size:** 40

---

##  Training Results

| Metric | Value |
|---|---|
| Final Training Loss | 2.33e-04 |
| Final Validation Loss | 1.1668 |
| Epochs | 25 |

Training and validation loss curves were plotted to monitor for overfitting.

---

##  Gradio Web Interface

```python
import gradio as gr

interface = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="krishna brain tumor classification",
    description="Brain tumor app. Let's learn!",
    article="Select an image or upload one to predict if brain tumor is present or not",
    theme="Glass",
    examples=[...]   # 2 sample images from dataset
)

interface.launch(share=True, debug=True)
```

---

##  Setup & Installation

### Prerequisites

```bash
pip install tensorflow keras gradio numpy pillow scikit-learn matplotlib
```

### Running in Google Colab

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Set dataset path:
```python
DATADIR = '/content/drive/MyDrive/brain_tumor_dataset/'
```

3. Train the model or load a saved model:
```python
model = keras.models.load_model('brain_tumor_model.h5')
```

4. Launch the Gradio app:
```python
interface.launch(share=True, debug=True)
```

A **public shareable link** will be generated (valid for 1 week).

---

##  Inference

```python
from PIL import Image
import numpy as np

def recognize_image(image):
    img = Image.fromarray(image).resize((128, 128))
    x = np.array(img).reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    return names(classification)

def names(number):
    if number == 0:
        return "It's a Tumor"
    else:
        return "No, It's not a Tumor"
```

### Example Predictions

| Image | Confidence | Result |
|---|---|---|
| `no/N16.jpg` | 100.0% | No, It's not a Tumor |
| `yes/Y11.jpg` | 99.9999% | It's a Tumor |

---

##  Project Structure

```
brain_tumor_classification/
├── brain_tumor_dataset/
│   ├── yes/          # MRI images with tumors
│   └── no/           # MRI images without tumors
├── brain_tumor_cnn.ipynb   # Main Jupyter notebook
├── brain_tumor_model.h5    # Saved model (after training)
└── README.md
```

---

 Known Issues

- Gradio may throw `asyncio` event loop errors in some Colab environments. If this occurs, restart the runtime and re-run the launch cell.
- The public Gradio link expires after **1 week**.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

