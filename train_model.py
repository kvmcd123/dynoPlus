from utility.modelHW import modelHW
import pickle
import os

# Model and Dataset Parameters
dataset_class       = 'inverter_data'            # Class for model dataset
dataset_name        = 'test_full'            # Folder name for model dataset
model_name          = 'model_full'  # Model Name

# Sensitivity Parameters
sens_limit = 0.01
sens_weight = 0.2
sens_delta = 0.01
sens_input = ['Vd','Vq']
sens_output = ['Isd','Isq']
sens_sets = {
    "Ud":   [0.3,0.9],
    "Uq":   [-0.15,0.15],
    "Vd":   [150,300],
    "Vq":   [150,300],
    # "Ra":   [6.0,59.0],
    # "Rb":   [6.0,59.0],
    # "Rc":   [6.0,59.0]
    }


# Class Initialization
dynoPlus = modelHW()

# Initialize the data parameters
dynoPlus.add_data_parameters(
    dataset_class,          # Class for model dataset
    dataset_name,           # Folder name for model dataset
    model_name,             # Model Name
    batch_size=128          # Batch Size
)

# Initialize the model parameters and create the model
dynoPlus.add_model_parameters(
    f1_output_dim   = 2,            # Input Dimensions of Linear Block
    f2_input_dim    = 2,            # Output Dimensions of Linear Block
    f1_nodes        = 20,           # Number of Input Hidden Nodes
    f2_nodes        = 20,           # Number of Output Hidden Nodes
    nb              = 3,            # Order of Linear Block 
    na              = 2,            # Order of Linear Block
    nk              = 0,            # Number of Time Delays
    activation      ='tanh',        # Activation Function for Nonlinearities
    model_type      ="2ndOrderX"    # Type of Model Structure
    )

dynoPlus.add_sensitivity_parameters(
    limit       = sens_limit,
    weight      = sens_weight,
    delta       = sens_delta,
    inputs      = sens_input,
    outputs     = sens_output,
    sets        = sens_sets
)

#Initialize the training parameters
dynoPlus.initialize_training_parameters(
    num_epochs=200,
    sensitivity_flag=True
)

# Train the model
dynoPlus.train()

# Save the model
dynoPlus.exportModel()

# Evaluate the trained model
dynoPlus.evaluate(8511, validation=False)

# Calculate the total rmse errors of the model
dynoPlus.calculate_total_rmse_errors()

# Specify the passivity parameters of the model
passivity_inputs = ['Vd','Vq']
passivity_epochs = 100

# Initialize the passivity parameters of the model
dynoPlus.initialize_passivity_parameters(passivity_inputs,passivity_epochs)

# Create the initial passivity input to test
u_initial = dynoPlus.u_normalized[dynoPlus.valid_trials[0]].to(dynoPlus.device)

# Calculat ethe passivity index
dynoPlus.find_passivity_type(u_initial)

# Save the class instance to a file
with open(os.path.join('models',model_name,model_name+".pkl"), "wb") as file:
    pickle.dump(dynoPlus, file)