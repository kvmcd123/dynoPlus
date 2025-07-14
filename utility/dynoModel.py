import torch
import os
import torch
import torch.nn as nn
from utility.static import MimoStaticNonLinearity
from utility.lti import MimoLinearDynamicalOperator, StableSecondOrderMimoLinearDynamicalOperatorX

# Define the Hammerstein-Wiener-like model
class dynoModel(nn.Module):
    def __init__(self, f1_input_dim, f1_hidden_dim, f1_output_dim,
                 g1_nb, g1_na, g1_nk,
                 f2_input_dim, f2_hidden_dim, f2_output_dim,activation,model_type):
        super(dynoModel, self).__init__()
        
        self.model_type = model_type
        
        if self.model_type == "Standard":
            # Configure the HW Block Operators
            self.F1 = MimoStaticNonLinearity(f1_input_dim, f1_output_dim, f1_hidden_dim, activation)  
            self.G1 = MimoLinearDynamicalOperator(f1_output_dim, f2_input_dim, g1_nb, g1_na, g1_nk)
            self.F2 = MimoStaticNonLinearity(f2_input_dim, f2_output_dim, f2_hidden_dim, activation)
        elif self.model_type == "2ndOrderX":
            # Configure the HW Block Operators
            self.F1 = MimoStaticNonLinearity(f1_input_dim, f1_output_dim, f1_hidden_dim, activation)  
            self.G1 = StableSecondOrderMimoLinearDynamicalOperatorX(f1_output_dim, f2_input_dim)
            self.F2 = MimoStaticNonLinearity(f2_input_dim, f2_output_dim, f2_hidden_dim, activation)
        
    def forward(self, u_in, y0=None, u0=None):
        if self.model_type == "Standard":
            # Input Nonlinearity
            y_1_nl = self.F1(u_in)
            # Linear System
            y_1_lin = self.G1(y_1_nl)
            # Output Nonlinearity
            y_hat = self.F2(y_1_lin)

        elif self.model_type == "2ndOrderX":
            # Input Nonlinearity
            y_1_nl = self.F1(u_in)
            # Linear System
            y_1_lin = self.G1(y_1_nl,y0,u0)
            # Output Nonlinearity
            y_hat = self.F2(y_1_lin)
        
        return y_hat,y_1_lin,y_1_nl

