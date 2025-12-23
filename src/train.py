import os
from src.generator import HomoglyphDataGenerator
from src.models import build_siamese_model, contrastive_loss
import src.config as config

# --- CONFIGURATION ---
# Change this to 'mobilenet' to use the newer model!
MODEL_TYPE = 'custom'  # Options: 'custom' or 'mobilenet'

def main():
    print(f"Initializing Data Generator (Batch Size: {config.BATCH_SIZE})...")
    train_gen = HomoglyphDataGenerator()
    
    # Build the selected model
    model = build_siamese_model(model_type=MODEL_TYPE)
    
    model.compile(loss=contrastive_loss, optimizer='adam')
    model.summary()
    
    print(f"Starting Training ({config.EPOCHS} Epochs)...")
    model.fit(train_gen, epochs=config.EPOCHS)
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    # Save with a specific name so we know which one it is
    save_path = f'saved_models/siamese_{MODEL_TYPE}.keras'
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
