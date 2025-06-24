
from utility.modelHW import modelHW
######################################################################################
#                       DATSET DESCRIPTION    
######################################################################################
description = '''
This dataset consists of different step, sinusoidal, ramp, and random noise input output
responses for identifying a simple second order nonlinear model.
'''
######################################################################################
#
######################################################################################


######################################################################################
#                       DATSET CREATION    
######################################################################################

# Initiate the modelHW class
dynoPlus = modelHW()

# Create the new dataset with the specified parameters
dynoPlus.create_dataset(
    dataset_class               = 'test',                           # Name of dataset folder
    dataset_name                = 'test' ,                          # Name of dataset
    simulation_time             = 1,                                # Total simulation time in seconds
    raw_data_filename           = 'raw_data/testResults_11.mat',    # Name of Experimental Data File
    exclude_input_variables     = ['U0','V0'],                      # Input Variables in Data File to be Excluded
    exclude_output_variables    = ['Is0'],                          # Output Variables in Data File to be Excluded
    description                 = description                       # Dataset Description
)
