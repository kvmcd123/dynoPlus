from utility.modelHW import modelHW
import pickle
import os

# Model and Dataset Parameters
dataset_class       = 'test'            # Class for model dataset
dataset_name        = 'test'            # Folder name for model dataset
model_name          = 'new_model_test'  # Model Name

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
    f1_output_dim   = 2,        # Input Dimensions of Linear Block
    f2_input_dim    = 2,        # Output Dimensions of Linear Block
    nb              = 2,        # Order of Linear Block 
    na              = 2,        # Order of Linear Block
    nk              = 0,        # Number of Time Delays
    activation      ='tanh'     # Activation Function for Nonlinearities
    )

#Initialize the training parameters
dynoPlus.initialize_training_parameters(num_epochs=2)

# Train the model
dynoPlus.train()

# Save the model
dynoPlus.exportModel()

# Evaluate the trained model
dynoPlus.evaluate(67)

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