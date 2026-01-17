# 🕵️‍♂️ Deep Learning Homoglyph Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-State%20of%20the%20Art-red)

A State-of-the-Art (SOTA) visual phishing detection system. This project uses **Siamese Convolutional Neural Networks (CNNs)** to identify homoglyph attacks—phishing domains that look visually identical to legitimate brands (e.g., `goog1e.com` vs `google.com`) but use different characters.

Unlike traditional regex or blocklist-based filters, this model **"sees"** the URL just like a human does, enabling **Zero-Shot detection** of new, unseen phishing attacks.

---

## 📸 Demo & Results

The model calculates the **Visual Euclidean Distance** between two rendered text strings.
* **Low Distance (< 0.5):** Visually confusing (High Phishing Risk).
* **High Distance (> 0.5):** Visually distinct (Safe).

**Example Output:**
* **Target:** `microsoft.com`
* **Attack:** `rn˛ｃ𝐫ంｓం𝐟𝐭𝅭ｃంrn` (Complex Unicode Spoof)
* **Result:** **Detected** (Visual Distance: 0.12)

---

## 🌟 Key Features

* **👁️ Visual Semantic Similarity:** Uses a Siamese Network architecture to learn the *visual* features of characters rather than their ASCII codes.
* **♾️ Infinite Synthetic Data:** Includes a custom `HomoglyphDataGenerator` that generates infinite training pairs on-the-fly using randomized fonts, noise, kerning, and sizes. This prevents overfitting to specific strings.
* **🚀 Zero-Shot Learning:** Can protect brands it was never trained on. If you train it on Google, it can still protect PayPal because it understands *visual similarity* universally.
* **🧠 Architecture Comparison:** Includes implementations of both a lightweight **Custom CNN** and a transfer-learning based **MobileNetV2** backbone.
* **🛡️ Real-World Validation:** Validated against the `confusable_homoglyphs` library to ensure robustness against actual IDN (Internationalized Domain Name) attacks.

---

## 🧠 Model Architecture

The project implements a **Siamese Network** with **Contrastive Loss**.

```text
[Input Image A] ----> [Shared CNN Backbone] ----> [Feature Vector A]
                                                         |
                                                    [Euclidean Distance] ----> Similarity Score (0-1)
                                                         |
[Input Image B] ----> [Shared CNN Backbone] ----> [Feature Vector B]

```

1. **Inputs:** Two grayscale images (128x64) containing the target URL and the suspect URL.
2. **Backbone:** A shared Convolutional Neural Network (weights are tied) extracts high-level visual features.
3. **Distance:** The model calculates the geometric distance between the two feature vectors.

---

## 📂 Project Structure

```text
homoglyph-detection/
│
├── notebooks/
│   └── SOTA_Homoglyph_Experiments.ipynb  # Full analysis, visualization, and experiments
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Hyperparameters (Batch size, Img size)
│   ├── generator.py        # Dynamic Synthetic Data Generator
│   ├── models.py           # Siamese Network & MobileNetV2 definitions
│   ├── train.py            # Training script
│   ├── inference.py        # CLI tool for checking domains
│   └── utils.py            # Font downloading and helper functions
│
├── saved_models/           # Stores trained model weights (.keras)
├── requirements.txt        
└── README.md               

```

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/AkshitSingh7/homoglyph-detection-dl.git](https://github.com/AkshitSingh7/homoglyph-detection-dl.git)
cd homoglyph-detection-dl

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## 🚀 Usage

### 1. Training the Model

To train the Siamese network from scratch using the synthetic generator:

```bash
python -m src.train

```

*This will save the model to `saved_models/siamese_homoglyph.keras`.*

### 2. Running Inference (The Demo)

To test the model on specific domain pairs (e.g., Microsoft vs. a Unicode spoof):

```bash
python -m src.inference

```

### 3. Using in Your Code

You can use the trained model in your own Python scripts:

```python
from src.inference import predict_similarity
from src.generator import HomoglyphDataGenerator
import tensorflow as tf

# Load Model
model = tf.keras.models.load_model('saved_models/siamese_homoglyph.keras')
gen = HomoglyphDataGenerator()

# Check a pair
distance, _, _ = predict_similarity(model, gen, "paypal.com", "pɑypal.com")

if distance < 0.5:
    print(f"🚨 PHISHING ALERT! Visual Distance: {distance:.4f}")
else:
    print(f"Safe. Distance: {distance:.4f}")

```

---

## 📊 Performance & Nuance

Why is this better than **Levenshtein Distance**?

* **Levenshtein (Edit Distance):**
* `google` vs `goog1e` -> Distance 1 (Close)
* `google` vs `gOOgle` -> Distance 2 (Farther)
* *Result:* Fails to capture that `gOOgle` is visually harmless, while `goog1e` is a trap.


* **Our Deep Learning Approach:**
* `google` vs `goog1e` -> **Visual Distance 0.1** (Alert!)
* `google` vs `facebook` -> **Visual Distance 1.5** (Safe)
* *Result:* Captures the **human perception** of the text.



---

## 🤝 Contributing

Contributions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
