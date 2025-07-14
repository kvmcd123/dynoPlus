import pickle
import os
import addcopyfighandler
# Specify the name of the model you want to run
model_name = 'new_model_no_r_sens'

# Load the model back
with open(os.path.join('models',model_name,model_name+".pkl"), "rb") as file:
    dynoPlus = pickle.load(file)

# Initialize the model class again
dynoPlus.initialize()

# Evaluate a test case of the model
dynoPlus.evaluate(67)

# Calculate the total rmse errors of the model
dynoPlus.calculate_total_rmse_errors()