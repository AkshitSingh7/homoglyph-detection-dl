import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import src.config as config
from src.utils import get_font

class HomoglyphDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=config.BATCH_SIZE, steps=config.STEPS_PER_EPOCH, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.img_w = config.IMG_WIDTH
        self.img_h = config.IMG_HEIGHT
        self.steps = steps
        self.domains = config.ANCHOR_DOMAINS
        self.homoglyphs = {'o': '0', 'l': '1', 'i': '1', 'a': 'α', 'e': '3', 't': 'τ'}

    def __len__(self):
        return self.steps

    def render_text(self, text):
        img = Image.new('L', (self.img_w, self.img_h), color=255)
        draw = ImageDraw.Draw(img)
        
        font_size = random.randint(config.FONT_SIZE_MIN, config.FONT_SIZE_MAX)
        font = get_font(font_size)
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(text, font=font)
            
        x = (self.img_w - w) / 2 + random.randint(-5, 5)
        y = (self.img_h - h) / 2 + random.randint(-5, 5)
        
        draw.text((x, y), text, font=font, fill=0)
        
        np_img = np.array(img) / 255.0
        noise = np.random.normal(0, 0.05, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        return np_img.reshape(self.img_h, self.img_w, 1)

    def generate_fake(self, domain):
        chars = list(domain)
        for i, char in enumerate(chars):
            if char in self.homoglyphs and random.random() > 0.5:
                chars[i] = self.homoglyphs[char]
        return "".join(chars)

    def __getitem__(self, index):
        pairs_1 = []
        pairs_2 = []
        labels = []
        
        for _ in range(self.batch_size):
            anchor_text = random.choice(self.domains)
            
            if random.random() > 0.5:
                # Positive Pair (Same Visuals)
                target_text = anchor_text 
                label = 0.0 
            else:
                # Negative Pair (Different Visuals)
                if random.random() > 0.5:
                     target_text = self.generate_fake(anchor_text)
                else:
                     target_text = random.choice([d for d in self.domains if d != anchor_text])
                label = 1.0 
            
            pairs_1.append(self.render_text(anchor_text))
            pairs_2.append(self.render_text(target_text))
            labels.append(label)
            
        return (np.array(pairs_1), np.array(pairs_2)), np.array(labels)
