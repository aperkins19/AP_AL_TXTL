
MLP_Settings_Dictionary = {
# top layer is each MLP
# Second layer is each layer in the MLP

    
    "1":{
        
        "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 1
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }

    },

    "2":{
        
        "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 2
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },

    "3":{
        
        "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 3
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },

    "4":{
        
        "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 4
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },

    "5":{
        
        "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 5
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },

    "6":{
       "HyperParams": {
            
            "learning_rate": 0.001,
            "loss_function": "mse",
            "metrics": "mae",
            "RandomSeed": 6
        },

        "layers":{
            1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
            2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
            4: {"Hidden Nodes": 20, "activation":"relu"},
            5: {"Output": True, "activation":"relu"}
        }
    },


    "7":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 7
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
    }
},

    
    "8":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 8
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "9":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 9
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "10":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 10
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "11":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 11
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "12":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 12
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "13":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 13
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "14":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 14
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "15":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 15
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "16":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 16
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "17":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 17
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "18":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 18
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "19":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 19
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "20":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 20
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "21":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 21
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "22":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 22
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "23":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 23
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "24":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 24
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

    "25":{
    "HyperParams": {
        
        "learning_rate": 0.001,
        "loss_function": "mse",
        "metrics": "mae",
        "RandomSeed": 25
    },

    "layers":{
        1: {"Input":True,"Hidden Nodes": 10, "activation":"relu"},
        2: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        3: {"Hidden Nodes": 100, "activation":"relu", "Dropout": 0.5},
        4: {"Hidden Nodes": 20, "activation":"relu"},
        5: {"Output": True, "activation":"relu"}
     }
    },

}