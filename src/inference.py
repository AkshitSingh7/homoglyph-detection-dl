import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.generator import HomoglyphDataGenerator
from src.models import contrastive_loss
try:
    from confusable_homoglyphs import confusables
except ImportError:
    pass

def predict_similarity(model, gen, text1, text2):
    img1 = gen.render_text(text1)
    img2 = gen.render_text(text2)
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    dist = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    return dist, img1, img2

def run_demo():
    # Load Model
    print("Loading Model...")
    try:
        model = tf.keras.models.load_model(
            'saved_models/siamese_homoglyph.keras', 
            custom_objects={'contrastive_loss': contrastive_loss}
        )
    except:
        print("Model not found! Please run 'python -m src.train' first.")
        return

    gen = HomoglyphDataGenerator()
    
    target = "microsoft.com"
    # Create a nuanced fake
    fake = target.replace('o', 'ο').replace('i', '1') # Greek Omicron, Number 1

    print(f"Comparing {target} vs {fake}")
    dist, img1, img2 = predict_similarity(model, gen, target, fake)
    
    print(f"Visual Distance: {dist:.4f}")
    if dist < 0.5:
        print("🚨 ALERT: Homoglyph Detected!")
    else:
        print("Safe.")

    # Show Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1.squeeze(), cmap='gray')
    plt.title("Real")
    plt.subplot(1, 2, 2)
    plt.imshow(img2.squeeze(), cmap='gray')
    plt.title(f"Fake (Dist: {dist:.2f})")
    plt.show()

if __name__ == "__main__":
    run_demo()
