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

def build_base_cnn(input_shape):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(inputs, x)

def build_siamese_model(base_network=None):
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
    
    if base_network is None:
        base_network = build_base_cnn(input_shape)
        
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    model = Model([input_a, input_b], distance)
    return model
