import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

def define_model(input_nodes, num_output_nodes):

    model = keras.Sequential([
    keras.layers.Dense(32, input_shape = (input_nodes,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    #keras.layers.Dense(num_output_nodes, activation='linear')
    keras.layers.Dense(num_output_nodes)
    ])

    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss= "mse",
                metrics=['mae'])

    return model


def generate_MLP_ensemble():

    """"Iterates over the """

    MLP_ensemble = []

    return MLP_ensemble