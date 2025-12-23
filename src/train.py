import os
from src.generator import HomoglyphDataGenerator
from src.models import build_siamese_model, contrastive_loss
import src.config as config

def main():
    print("Initializing Data Generator...")
    train_gen = HomoglyphDataGenerator()
    
    print("Building Siamese Network...")
    model = build_siamese_model()
    model.compile(loss=contrastive_loss, optimizer='adam')
    
    print("Starting Training...")
    model.fit(train_gen, epochs=config.EPOCHS)
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    model.save('saved_models/siamese_homoglyph.keras')
    print("Model saved to saved_models/siamese_homoglyph.keras")

if __name__ == "__main__":
    main()
