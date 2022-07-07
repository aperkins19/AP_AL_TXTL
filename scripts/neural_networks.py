import tensorflow as tf
from tensorflow import keras

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

    
    model.compile(optimizer='adam',
                loss= "mse",
                metrics=['mae'])

    return model