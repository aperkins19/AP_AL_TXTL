import tensorflow as tf
from tensorflow import keras

def define_model(input_nodes, num_output_nodes):

    model = keras.Sequential([
    keras.layers.Dense(32, input_shape = (5,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_output_nodes, activation='softmax')
    ])

    return model