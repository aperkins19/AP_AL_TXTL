
MLP_Settings_Dictionary = {
# top layer is each MLP
# Second layer is each layer in the MLP

    
    "1":{
        
        "HyperParams": {
            
            "learning_rate": 0.0001,
            "loss_function": "mse",
            "metrics": "mae"
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 32, "activation":"relu"},
            2: {"Hidden Nodes": 128, "activation":"relu", "Dropout": 0.2},
            3: {"Hidden Nodes": 128, "activation":"relu", "Dropout": 0.2},
            4: {"Hidden Nodes": 32, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }

    },

    "2":{
        
        "HyperParams": {
            
            "learning_rate": 0.0001,
            "loss_function": "mse",
            "metrics": "mae"
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 32, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.2},
            3: {"Hidden Nodes": 128, "activation":"relu", "Dropout": 0.2},
            4: {"Hidden Nodes": 32, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },

    "3":{
        
        "HyperParams": {
            
            "learning_rate": 0.0001,
            "loss_function": "mse",
            "metrics": "mae"
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 32, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.2},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.2},
            4: {"Hidden Nodes": 32, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },
}