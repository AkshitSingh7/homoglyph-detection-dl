import tensorflow as tf
from tensorflow.keras import layers, models, Model, applications
import tensorflow.keras.backend as K
import src.config as config

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- MODEL 1: Custom Lightweight CNN (The "Old" Model) ---
def build_base_cnn(input_shape):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(inputs, x, name="Custom_CNN_Backbone")

# --- MODEL 2: MobileNetV2 (The "New" SOTA Model) ---
def build_mobilenet_model(input_shape):
    inputs = layers.Input(input_shape)
    
    # MobileNet needs 3 channels (RGB), but we have 1 (Grayscale).
    # We add a generic Conv2D layer to project 1 channel -> 3 channels.
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
    
    # Load MobileNetV2 (Pre-trained on ImageNet)
    # We define input_shape as (H, W, 3) because we just converted it.
    base_model = applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze weights for faster training (Optional)
    base_model.trainable = False
    
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    
    return Model(inputs, x, name="MobileNet_Backbone")

# --- SIAMESE BUILDER ---
def build_siamese_model(model_type='custom'):
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
    
    # Select Backbone
    if model_type == 'mobilenet':
        print(">> Building Siamese Network with MobileNetV2 Backbone...")
        base_network = build_mobilenet_model(input_shape)
    else:
        print(">> Building Siamese Network with Custom CNN Backbone...")
        base_network = build_base_cnn(input_shape)
        
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Share weights
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    model = Model([input_a, input_b], distance)
    return model
