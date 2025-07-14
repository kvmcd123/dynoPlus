import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings
from utility.dynoModel import dynoModel
import matplotlib.pyplot as plt
from utility.metrics import fit_index 
import h5py
import json
import mat73
from scipy import signal
from scipy.signal import hilbert

class modelHW():
    """
    Hardware model wrapper that handles global setup and instantiates
    the Dyno neural network model with the specified architecture.
    """
    def __init__(self):
        """
        Constructor.

        Calls the base class initializer and runs the common
        initialization routine.
        """
        super(modelHW, self).__init__()
        self.initialize()

    def initialize(self):
        """
        Global initialization routine.

        - Sets environment variables to avoid OpenMP duplicate-lib errors.
        - Suppresses non-critical warnings (e.g., plot formatter messages).
        - Seeds NumPy and PyTorch RNGs for reproducible experiments.
        - Configures NumPy to raise exceptions on any floating-point warning.
        """
        # Allow duplicate OpenMP library loads without crashing
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # Ignore warnings from matplotlib/other formatters
        warnings.filterwarnings("ignore")
        # Fix the random seed so experiments are repeatable
        np.random.seed(0)
        torch.manual_seed(0)
        # Turn floating-point warnings into errors for easier debugging
        np.seterr(all='raise')                      # Shows more details about overflow error
    
    def initialize_filter(self):
        self.bord = 6
        self.blev = 0.05
        self.flt_sos_lp = signal.butter (self.bord, self.blev, btype='lowpass', output='sos')
        self.flt_sos_hp = signal.butter (self.bord, self.blev, btype='highpass', output='sos')
    
    def moving_average(self, a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n
    
    def my_filter(self,a,hp=False):
        if hp:
            raw = np.abs(signal.sosfiltfilt (self.flt_sos_hp, a, padtype=None))
            avg = self.moving_average(raw, n=30)
            return signal.sosfiltfilt (self.flt_sos_lp, avg, padtype=None)
        else:
            return signal.sosfiltfilt (self.flt_sos_lp, a, padtype=None)
    
    
    
    def create_dataset(self,raw_data_filename, dataset_class,dataset_name, simulation_time, exclude_input_variables=None,exclude_output_variables=None,description=None):
        data_dict = mat73.loadmat(raw_data_filename)
        # Useful Stats
        numCases = len(data_dict['testResults']['inputs'])
        print(len(data_dict['testResults']['inputs'][0][0]))
        numInputs = len(data_dict['testResults']['inputs'][0][0])-len(exclude_input_variables) 
        numOutputs = len(data_dict['testResults']['outputs'][0][0])-len(exclude_output_variables) 


        # Load Input Variable Names and Create Lists
        inputVariables = []
        for key in data_dict['testResults']['inputs'][0][0]:
            if key not in exclude_input_variables:
                    inputVariables.append(key)
                    locals()[key] = []

        # Load Output Variable Names and Create Lists
        outputVariables = []
        for key in data_dict['testResults']['outputs'][0][0]:
            if key not in exclude_output_variables:
                outputVariables.append(key)
                locals()[key] = []

        unbalanced = False
        if "unbalanced" in data_dict['testResults']:
            if data_dict['testResults']['unbalanced'][0][0] == 1:
                unbalanced = True
                self.initialize_filter()

        # Initialize a dummy model to ensure inputs and outputs work
        print(unbalanced)
        self.add_model_parameters(
            f1_input_dim=numInputs,
            f2_output_dim=numOutputs,
            f1_output_dim=2,
            f2_input_dim=2, 
            nb=3,
            na=2,
            nk=0,
            activation='tanh')

        dataset_folder = os.path.join('datasets',dataset_class,dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        # Create the HDF5 file and open it
        with h5py.File(os.path.join(dataset_folder,dataset_name+'.h5'),'w') as hdf:
            
            # Create a group for test cases
            data_group = hdf.create_group('data')
            
            # Create a group for scaling factors
            scaling_group = hdf.create_group('scaling_factors')
            
            # Create a group for variables
            variables_group = hdf.create_group('variables')

            # Add the input and output variables to their respective groups
            variables_group.create_dataset(f'input_variables',data = inputVariables)
            variables_group.create_dataset(f'output_variables',data = outputVariables)

            # Process each test result
            for i in range(numCases):
                print('Processing Test Case ' + str(i))

                # Create the case group to store the input and output data
                case_group = data_group.create_group(f'case_{i}')

                # Grab the input output data for test case i
                inputs = data_dict['testResults']['inputs'][i][0]
                outputs = data_dict['testResults']['outputs'][i][0]
                time = np.linspace(0,1,len(inputs['Vd']))
                if unbalanced:
                    # Check if test case is unbalanced
                    unbalanced_flag = data_dict['testResults']['unbalanced'][i][0]
                    if unbalanced_flag == 1 :
                        Vd_temp = inputs['Vd']
                        Vq_temp = inputs['Vq']
                        V0_temp = inputs['V0']

                        Id_temp = outputs['Isd']
                        Iq_temp = outputs['Isq']
                        I0_temp = outputs['Is0']
                        
                        Vdlo = self.my_filter (Vd_temp, hp=False)
                        Vdhi = self.my_filter (Vd_temp, hp=True)
                        Vqlo = self.my_filter (Vq_temp, hp=False)
                        Vqhi = self.my_filter (Vq_temp, hp=True)
                        V0lo = self.my_filter (V0_temp, hp=False)
                        V0hi = self.my_filter (V0_temp, hp=True)

                        Idhi = self.my_filter (Id_temp, hp=True)
                        Idlo = self.my_filter (Id_temp, hp=False)
                        Iqlo = self.my_filter (Iq_temp, hp=False)
                        Iqhi = self.my_filter (Iq_temp, hp=True)
                        I0lo = self.my_filter (I0_temp, hp=False)
                        I0hi = self.my_filter (I0_temp, hp=True)

                        wc = 2*np.pi*60
                        
                        outputs['Isd'] = Idlo + Idhi * np.sin(2*wc*time)
                        outputs['Isq'] = Iqlo + Iqhi * np.sin(2*wc*time)
                        outputs['Is0'] = I0lo + I0hi * np.sin(wc*time)

                        inputs['Vd'] = Vdlo + Vdhi * np.sin(2*wc*time)
                        inputs['Vq'] = Vqlo + Vqhi * np.sin(2*wc*time)
                        inputs['V0'] = V0lo + V0hi * np.sin(wc*time)

                    # If test case is unbalanced, filter I and V components and add 
                # Convert input and output data directly to torch tensors
                input_data = torch.tensor(
                    [inputs[var] for var in inputVariables], dtype=torch.float
                ).transpose(0, 1).unsqueeze(0)

                output_data = torch.tensor(
                    [outputs[var] for var in outputVariables], dtype=torch.float
                ).transpose(0, 1).unsqueeze(0)

                # Test the model works with data
                #y_hat,_,_ = self.dynoModel(input_data)
                y_hat,_,_ = self.dynoModel(input_data,self.y0,self.u0)
                
                case_group.create_dataset('input', data=input_data)
                case_group.create_dataset('output', data=output_data)

                for j in range(numInputs):
                    locals()[inputVariables[j]].append(input_data[:,:,j])

                for j in range(numOutputs):
                    locals()[outputVariables[j]].append(output_data[:,:,j])

            for key in inputVariables:
                print(np.mean(locals()[key]))
                print(np.ptp(locals()[key]) or 1)
                variable_group = scaling_group.create_group(key)
                variable_group.create_dataset(f'mean',data = np.mean(locals()[key]))
                variable_group.create_dataset(f'range',data = np.ptp(locals()[key]) or 1.0)

            for key in outputVariables:
                variable_group = scaling_group.create_group(key)
                print(float(np.mean(locals()[key])))
                print(float(np.ptp(locals()[key])))
                variable_group.create_dataset(f'mean',data = float(np.mean(locals()[key])))
                variable_group.create_dataset(f'range',data = float(np.ptp(locals()[key])))
        
        if description:
            with open(os.path.join(dataset_folder,"description.txt"), "w") as text_file:
                text_file.write("%s" % description)



    def initialize_dynoModel(self):
        """
        Instantiate the dynoNet neural network.

        Uses the following hyperparameters that should have been
        set on self beforehand:
          f1_input_dim   - number of inputs to stage 1
          f1_nodes       - hidden nodes in stage 1
          f1_output_dim  - outputs from stage 1
          nb, na, nk     - dynamic model orders and delay
          f2_input_dim   - inputs to stage 2
          f2_nodes       - hidden nodes in stage 2
          f2_output_dim  - outputs from stage 2
          activation     - activation function for both stages
          model_type     - the type of model structure to be used
        """
        # Build the DynoModel using stored architecture parameters
        self.dynoModel = dynoModel(
            self.f1_input_dim, 
            self.f1_nodes, 
            self.f1_output_dim, 
            self.nb, 
            self.na, 
            self.nk, 
            self.f2_input_dim, 
            self.f2_nodes, 
            self.f2_output_dim, 
            self.activation,
            self.model_type
            )
    
    #def add_data_parameters(self,dataset_class,dataset_name, model_name,f1_output_dim,na,nb,nk,f2_input_dim,f1_nodes = 20, f2_nodes = 20,activation='tanh', dt=0.01,batch_size=12,device = "cpu"):
    def add_data_parameters(self,dataset_class,dataset_name, model_name, dt=0.01,batch_size=12,device = "cpu"):
        """
        Configure dataset paths, load variable definitions, set model hyperparameters,
        and initialize the dynoNet and training data processing.

        Parameters
        ----------
        dataset_class : str
            Top-level category name of the dataset (e.g., 'industrial', 'biomedical').
        dataset_name : str
            Specific dataset identifier (corresponds to HDF5 file and folder name).
        model_name : str
            User-friendly name for this model configuration.
        f1_output_dim : int
            Number of outputs from the first neural stage.
        na : int
            Number of autoregressive terms in the dynamic model.
        nb : int
            Number of input terms in the dynamic model.
        nk : int
            Input delay (dead-time) in the dynamic model.
        f2_input_dim : int
            Number of inputs to the second neural stage.
        f1_nodes : int, optional
            Hidden layer size for the first neural stage (default=20).
        f2_nodes : int, optional
            Hidden layer size for the second neural stage (default=20).
        activation : str, optional
            Activation function name for both stages (default='tanh').
        dt : float, optional
            Sampling time interval for the dynamic system (seconds) (default=0.01).
        batch_size : int, optional
            Mini-batch size for training (default=12).
        device : str, optional
            Compute device ("cpu" or "cuda") for model and data (default="cpu").
        """
        
        # Store dataset and model identifiers
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.model_name = model_name

        # Build path to the dataset folder and open its HDF5 file
        self.dataset_folder = os.path.join('datasets',dataset_class,dataset_name)
        hdf = h5py.File(os.path.join(self.dataset_folder,dataset_name+'.h5'),'r')
        
        # Read input/output variable names from the HDF5 file
        self.input_variables = hdf['variables/input_variables'][:].astype('str')
        self.output_variables = hdf['variables/output_variables'][:].astype('str')

        # Store time step, and training batch size
        self.dt = dt
        self.batch_size=batch_size
                
        # Store device
        self.device = device 

        # Prepare and preprocess training data from the HDF5 file
        self.process_training_data(hdf)



    def add_model_parameters(self,f1_output_dim,f2_input_dim, nb,na,nk,f1_nodes=20,f2_nodes=20,f1_input_dim=None,f2_output_dim=None,activation='tanh',model_type = "Standard"):
        
        if f1_input_dim:
            self.f1_input_dim=f1_input_dim
        else:
            self.f1_input_dim = len(self.input_variables)

        if f2_output_dim:
            self.f2_output_dim=f2_output_dim
        else:
            self.f2_output_dim = len(self.output_variables)

        # Derive stage dimensions from variable counts and provided args
        self.f1_output_dim = f1_output_dim
        self.f2_input_dim = f2_input_dim

        # Store dynamic model orders and neural network widths
        self.nb = nb
        self.na = na
        self.nk = nk
        self.f1_nodes = f1_nodes
        self.f2_nodes = f2_nodes
        
        if hasattr(self, 'batch_size'):
            self.y0 = torch.zeros((self.batch_size, self.na), dtype=torch.float)
            self.u0 = torch.zeros((self.batch_size, self.nb), dtype=torch.float)
        else:
            self.y0 = torch.zeros((1, self.na), dtype=torch.float)
            self.u0 = torch.zeros((1, self.nb), dtype=torch.float)
        
        self.activation = activation
        
        self.model_type = model_type
        
        # Instantiate the DynoModel with the configured parameters
        self.initialize_dynoModel()

    def setup_optimizer(self):
        """
        Create and configure the optimizer for training.

        Dynamically looks up the optimizer class in torch.optim by name
        and instantiates it with the model parameters and hyperparameters
        that have been set on self (lr, eps, weight_decay).
        """
        # Resolve optimizer class (e.g., torch.optim.Adam) from its name
        OptimizerClass = getattr(optim, self.optimizer_name)
        
        # Instantiate with model parameters and optimizer-specific settings
        self.optimizer = OptimizerClass(self.dynoModel.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)

    def initialize_training_parameters(self,optimizer="Adam",criterion="MSE",num_epochs=1000,device = "cpu",msg_freq=100,dc_gain_flag=False,lr =0.001,eps=0,weight_decay=0,sensitivity_flag=False):
        """
        Set up core training hyperparameters, loss function, and optimizer.

        Parameters
        ----------
        optimizer : str, optional
            Name of the torch.optim optimizer to use (default="Adam").
        criterion : str, optional
            Loss function identifier ("MSE" supported) (default="MSE").
        num_epochs : int, optional
            Total number of training epochs (default=1000).
        device : str, optional
            Compute device ("cpu" or "cuda") for model and data (default="cpu").
        msg_freq : int, optional
            How often (in epochs) to print progress messages (default=100).
        dc_gain_flag : bool, optional
            If True, log or enforce DC gain constraints during training (default=False).
        lr : float, optional
            Learning rate for the optimizer (default=0.001).
        eps : float, optional
            Epsilon value for numerical stability in optimizer (default=0).
        weight_decay : float, optional
            L2 regularization weight decay term (default=0).
        """
        
        # Store optimizer hyperparameters
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Configure the loss function
        self.criterion_name = criterion
        if self.criterion_name == "MSE":
            # Mean Squared Error loss for regression
            self.criterion=nn.MSELoss()
        
        # Store remaining training settings
        self.num_epochs = num_epochs
        self.device = device
        self.msg_freq = msg_freq
        self.dc_gain_flag = dc_gain_flag
        self.sensitivity_flag = sensitivity_flag
        
        # Initialize loss history containers
        self.tLOSS = []     # training loss per epoch
        self.vLOSS = []     # validation loss per epoch (if used)
        
        # Instantiate optimizer with current settings
        self.setup_optimizer()

    def normalize(self, val,offset,scale):
        """
        Normalize data by subtracting an offset and dividing by a scale.

        Parameters
        ----------
        val : float or array-like
            Original value(s) to normalize.
        offset : float or array-like, same shape as val
            Value(s) to subtract from val (e.g., mean or baseline).
        scale : float or array-like, same shape as val
            Value(s) to divide into (e.g., range or standard deviation).

        Returns
        -------
        normalized : float or array-like
            Result of (val - offset) / scale.
        """
        # Shift data by offset, then scale to unit rang
        return (val-offset)/scale

    def denormalize(self, val,offset,scale):
        """
        Recover original data from normalized values.

        Parameters
        ----------
        val : float or array-like
            Normalized value(s) to invert.
        offset : float or array-like, same shape as val
            Original offset used during normalization.
        scale : float or array-like, same shape as val
            Original scale used during normalization.

        Returns
        -------
        original : float or array-like
            Result of val * scale + offset.
        """
        # Rescale normalized data and add back the offset
        return val*scale + offset
    
    def generate_scaling_factors(self,scaling_group):
        """
        Extract mean and range values for each variable from the provided group
        and store them as floats for later normalization/denormalization.

        Parameters
        ----------
        scaling_group : dict-like or h5py.Group
            Mapping of variable names to a sub-group or dict containing:
              - 'mean': array or scalar representing the variable's mean
              - 'range': array or scalar representing the variable's range

        Side Effects
        ------------
        Sets `self.scaling_factors` to a dict of the form:
            {
                variable_name: {
                    'mean': float(mean_value),
                    'range': float(range_value)
                },
                ...
            }
        """
        # Initialize the container for all variable scaling info
        self.scaling_factors = {}
        
        # Iterate through each variable in the group
        for variable in scaling_group:
            
            # Read raw mean and range (may be numpy or HDF5 scalar)
            mean = scaling_group[variable]['mean'][()]
            range_val = scaling_group[variable]['range'][()]
            
            # Store as native floats for downstream computations
            self.scaling_factors[variable] = {
                'mean': float(mean), 
                'range': float(range_val)
                }
        

    def process_training_data(self,hdf):
        """
        Load, normalize, and split training/validation datasets from HDF5.

        Parameters
        ----------
        hdf : h5py.File
            Open HDF5 file containing:
              - 'data': group with subgroups 'case_i' for each trial,
                         each containing 'input' and 'output' datasets.
              - 'scaling_factors': group with per-variable 'mean' and 'range'.

        Side Effects
        ------------
        - Generates scaling factors via `self.generate_scaling_factors`.
        - Builds and stores:
            self.num_datasets      : total number of trials
            self.num_valid_sets    : number of trials reserved for validation
            self.valid_trials      : list of trial indices for validation
            self.u_tensors, self.y_tensors            : raw input/output tensors
            self.u_normalized, self.y_normalized      : normalized tensors
            self.datasets_t, self.datasets_v          : lists of TensorDataset
            self.dataset_train, self.dataset_valid    : ConcatDataset splits
            self.validation_scale : train/validation size ratio
            self.train_dl, self.valid_dl              : DataLoader objects
        """
        # Access the data and scaling subgroups
        data_group = hdf['data']
        scaling_group = hdf['scaling_factors']

        # Build scaling factors lookup (mean & range per variable)
        self.generate_scaling_factors(scaling_group)
        
        # Determine total number of trials and size of validation set (~20%)
        self.num_datasets = len(data_group.items())
        self.num_valid_sets = round(0.2*self.num_datasets)#38
    
        # Randomly select unique trial indices for validation
        self.valid_trials = [np.random.randint(0,self.num_datasets-1) for _ in range(self.num_valid_sets)]

        # Initialize dictionaries to store data
        self.u_tensors = {} 
        self.y_tensors = {}
        self.u_normalized = {} 
        self.y_normalized = {}

        # Dictionary to store TensorDatasets
        self.datasets_v = []
        self.datasets_t = []
        self.datasets_total = []

        # Iterate through each trial case
        for i in range(self.num_datasets):
            # Load the input and output torch tensors
            self.u_tensors[i] = torch.tensor(data_group[f'case_{i}']['input'][:])
            self.y_tensors[i] = torch.tensor(data_group[f'case_{i}']['output'][:])
            
            # Normalize the input channels
            self.u_normalized[i] = self.u_tensors[i].clone()  # Clone to avoid modifying original tensors
            for j, key in enumerate(self.input_variables):
                mean = float(self.scaling_factors[key]['mean'])
                range_val = float(self.scaling_factors[key]['range'])
                self.u_normalized[i][0, :, j] = self.normalize(self.u_normalized[i][0, :, j], mean, range_val)

            # Normalize the output channels
            self.y_normalized[i] = self.y_tensors[i].clone()  # Clone to avoid modifying original tensors
            for j, key in enumerate(self.output_variables):
                mean = float(self.scaling_factors[key]['mean'])
                range_val = float(self.scaling_factors[key]['range'])
                self.y_normalized[i][0, :, j] = self.normalize(self.y_normalized[i][0, :, j], mean, range_val)

            # Determine if the ith trial should be used for training or validation
            dataset = torch.utils.data.TensorDataset(self.u_normalized[i], self.y_normalized[i])
            if i in self.valid_trials:
                self.datasets_v.append(dataset)
            else:
                self.datasets_t.append(dataset)

            self.datasets_total.append(dataset)

        # Concate the datasets from the list of datasets
        self.dataset_train = torch.utils.data.ConcatDataset(self.datasets_t)
        self.dataset_valid = torch.utils.data.ConcatDataset(self.datasets_v)
        self.dataset_total = torch.utils.data.ConcatDataset(self.datasets_total)

        # Compute scale factor for balancing loss metrics if needed
        self.validation_scale = float(len(self.dataset_train))/float(len(self.dataset_valid))

        # Instantiate PyTorch DataLoaders for batching
        self.train_dl = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batch_size, shuffle=True)
        self.valid_dl = torch.utils.data.DataLoader(self.dataset_valid, batch_size = self.batch_size, shuffle=False)
        self.total_dl = torch.utils.data.DataLoader(self.dataset_total, batch_size = 1, shuffle=False)
    
    def calculate_dc_loss(self,output,input):
        dc_gain = output/input
        loss_dc_gain = 0.001*torch.sqrt(torch.square(torch.mean(torch.sum(((0*dc_gain[:,-1,:])+1) - abs(dc_gain[:,-1,:])))))
        return loss_dc_gain
    
    def train(self):
        """
        Run the full training and validation loop for the DynoModel.

        This method performs the following steps:
        1. Iterates for `self.num_epochs` epochs.
        2. In each epoch:
           - Sets the model to training mode.
           - Loops over batches from `self.train_dl`:
             - Moves inputs/targets to `self.device`.
             - Zeros the optimizer gradients.
             - Performs a forward pass through `self.dynoModel`.
             - Computes the loss (optionally including DC‐gain regularization).
             - Accumulates `train_loss`, backpropagates, and updates weights.
           - Records the final batch loss in `self.tLOSS`.
           - Sets the model to evaluation mode and disables gradient tracking.
           - Loops over batches from `self.valid_dl`:
             - Moves data to device, forward pass, computes/accumulates validation loss.
           - Scales and records the validation loss in `self.vLOSS`.
        3. Prints progress markers and, at intervals (`self.msg_freq`) or on the last epoch,
           logs train/validation loss and RMSE to the console.
        4. After training completes, prints a completion message and calls `self.plot_training_loss()`
           to visualize the loss curves.

        Side Effects
        ------------
        - Updates `self.tLOSS` and `self.vLOSS` lists.
        - Prints progress and loss summaries to stdout.
        - Generates the training-loss plot via `self.plot_training_loss()`.
        """
        
        # Outer loop over epochs
        for epoch in range(self.num_epochs):

             # ---- TRAINING PHASE ----
            self.dynoModel.train()
            train_loss = 0.0
            
            # Iterate over training batches
            for ub,yb in self.train_dl:
                # Move batch to the configured device (CPU/GPU)
                ub, yb = ub.to(self.device), yb.to(self.device)  # Move data to GPU
                
                # Reset gradients before backprop
                self.optimizer.zero_grad()
                
                # Forward pass: get model outputs and any intermediate signals
                output, tf_output,tf_input = self.dynoModel(ub,self.u0,self.y0)
                
                # Compute loss, adding DC‐gain penalty if enabled
                loss = self.criterion(output, yb)
                
                if self.dc_gain_flag:
                    loss += self.calculate_dc_loss(tf_output,tf_input)
                
                if self.sensitivity_flag:
                    # loss = loss + sens_loss
                    loss += self.calculate_sensitivity()

                # Accumulate training loss for this epoch
                train_loss += loss.item()
                
                # Backpropagation and optimizer step
                loss.backward()
                
                self.optimizer.step()
            
            # Record the last batch loss in the training-loss history
            self.tLOSS.append(loss.item())
            
            # ---- VALIDATION PHASE ----
            self.dynoModel.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                # Iterate over validation batches
                for data, target in self.valid_dl:
                    
                    # Move data to GPU
                    data, target = data.to(self.device), target.to(self.device)  
                    
                    # Calculate the output
                    output, tf_output,tf_input = self.dynoModel(data,self.y0,self.u0)
                    
                    # Calculate the loss
                    loss = self.criterion(output,target)
                    if self.dc_gain_flag:
                        loss += self.calculate_dc_loss(tf_output,tf_input)
                    
                    if self.sensitivity_flag:
                        loss += self.calculate_sensitivity()
    
                    valid_loss += loss.item()
                
                # Apply scaling factor to normalize validation loss magnitude
                valid_loss *= self.validation_scale  
                
                # Record validation loss history
                self.vLOSS.append(valid_loss)

            # Provide visual update in console
            print('.', end ="", flush=True)

            # Prints training progress
            if (epoch % self.msg_freq == 0) or (epoch == self.num_epochs-1):
                print(f'\nIter {epoch+1} | Train Loss {train_loss:.6f} | Validation Loss {valid_loss:.6f} | Train RMSE: {np.sqrt(train_loss):.6f} | Valid RMSE: {np.sqrt(valid_loss):.6f}|')
        
        # Final message and loss-curve plotting
        print("Training complete!\n")
        self.plot_training_loss()

    def evaluate(self,n,validation=False):
        if validation == True:
            y_hat_test,_,_ = self.dynoModel(self.u_normalized[self.valid_trials[n]].to(self.device),self.y0,self.u0)
            y_valid_1=self.y_tensors[self.valid_trials[n]].detach().numpy()[0, :, :]
        else:
            y_hat_test,_,_ = self.dynoModel(self.u_normalized[n].to(self.device),self.y0,self.u0)
            y_valid_1=self.y_tensors[n].detach().numpy()[0, :, :]

        # Convert model output to numpy array
        y_hat_test_1 = y_hat_test.detach().cpu().numpy()[0, :, :]

        # Create a copy for denormalization inorder to not affect original
        y_hat_test_2 = np.copy(y_hat_test_1)

        for i, key in enumerate(self.output_variables):
            y_hat_test_2[:,i] = self.denormalize(y_hat_test_2[:,i],float(self.scaling_factors[key]['mean']),float(self.scaling_factors[key]['range']))
            
        # Extracting measured and estimated values
        measured = {var: y_valid_1[:, i] for i, var in enumerate(self.output_variables)}
        estimated = {var: y_hat_test_2[:, i] for i, var in enumerate(self.output_variables)}
        # Accuracy Calculations
        for var in self.output_variables:
            fit = fit_index(measured[var], estimated[var])
            print(f"{var} Accuracy: {fit.item():.2f}%")

        self.plot_evaluation(measured,estimated)

        # return measured, estimated

    def exportModel(self):
        """Writes the trained model coefficients to a JSON file.

        Includes all the training configuration parameters, and both continuous-time
        and discrete-time variants of *H1*.

        Args:
        filename (str): path and file name to the exported JSON file; typically ends in *_fhf.json*
        """
        if self.model_type == "Standard":
            a = self.dynoModel.G1.a_coeff.detach()
        elif self.model_type == "2ndOrderX":
            a_1 = 2.0 * torch.tanh(self.dynoModel.G1.alpha1)
            a_1_abs = torch.abs(a_1)
            a_2 = a_1_abs + (2.0 - a_1_abs) * torch.sigmoid(self.dynoModel.G1.alpha2) - 1.0
            a_coeff = torch.cat((a_1, a_2), dim=-1)
            a = a_coeff.detach()

        b = self.dynoModel.G1.b_coeff.detach()
        data = {}

        data['input_variables']     = self.input_variables.tolist()
        data['output_variables']    = self.output_variables.tolist()

        #  print ('a_coeff shape:', a.shape) # should be (n_out, n_in, n_a==n_b)
        for i in range(self.f1_output_dim):
            for j in range(self.f2_input_dim):
                key = 'a_{:d}_{:d}'.format(i, j)
                ary = a[i,j,:].numpy()
                data[key] = ary.tolist()

                key = 'b_{:d}_{:d}'.format(i, j)
                ary = b[i,j,:].numpy()
                data[key] = ary.tolist()

        block = self.dynoModel.F1.state_dict()
        n_in = self.dynoModel.F1.net[0].in_features
        n_hid = self.dynoModel.F1.net[0].out_features
        n_out = self.dynoModel.F1.net[2].out_features
        label = 'F1'
        activation = str(self.dynoModel.F1.net[1]).lower().replace("(", "").replace(")", "")
        data[label] = {'n_in': n_in, 'n_hid': n_hid, 'n_out': n_out, 'activation': activation}
        key = 'net.0.weight'
        data[label][key] = block[key][:,:].numpy().tolist()
        key = 'net.0.bias'
        data[label][key] = block[key][:].numpy().tolist()
        key = 'net.2.weight'
        data[label][key] = block[key][:,:].numpy().tolist()
        key = 'net.2.bias'
        data[label][key] = block[key][:].numpy().tolist()

        block = self.dynoModel.F2.state_dict()
        n_in = self.dynoModel.F2.net[0].in_features
        n_hid = self.dynoModel.F2.net[0].out_features
        n_out = self.dynoModel.F2.net[2].out_features
        
        label = 'F2'
        activation = str(self.dynoModel.F2.net[1]).lower().replace("(", "").replace(")", "")
        data[label] = {'n_in': n_in, 'n_hid': n_hid, 'n_out': n_out, 'activation': activation}
        key = 'net.0.weight'
        data[label][key] = block[key][:,:].numpy().tolist()
        key = 'net.0.bias'
        data[label][key] = block[key][:].numpy().tolist()
        key = 'net.2.weight'
        data[label][key] = block[key][:,:].numpy().tolist()
        key = 'net.2.bias'
        data[label][key] = block[key][:].numpy().tolist()

        # mean = offset
        # range = scale
        data['normfacs']={}
        for key in self.scaling_factors:
            data['normfacs'][key] = {'offset':float(self.scaling_factors[key]['mean']),
                                    'scale':float(self.scaling_factors[key]['range'])}

        data['dataset_class']       = self.dataset_class
        data['dataset_name']        = self.dataset_name
        data['model_name']          = self.model_name
        data['optimizer_name']      = self.optimizer_name
        data['lr']                  = self.lr
        data['eps']                 = self.eps
        data['weight_decay']        = self.weight_decay
        data['num_iter']            = self.num_epochs
        data['batch_size']          = self.batch_size
        data['n_validation_pct']    = 0.2
        data['na']                  = self.na
        data['nb']                  = self.nb
        data['nk']                  = self.nk
        data['activation']          = self.activation
        data['f1_nodes']            = self.f1_nodes
        data['f2_nodes']            = self.f2_nodes
        data['f1_output_dim']       = self.f1_output_dim
        data['f2_input_dim']        = self.f2_input_dim
        data['f2_output_dim']       = self.f2_output_dim
        data['criterion']           = self.criterion_name
        data['msg_freq']            = self.msg_freq
        data['dt']                  = self.dt
        data['dc_gain']             = self.dc_gain_flag
        data['sensitivity']         = self.sensitivity_flag

        
        file_path = os.path.join("models", self.model_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        modelPath = os.path.join(file_path, self.model_name)
        json_file_path = os.path.join(file_path, self.model_name + '.json')
        pt_file_path = os.path.join(file_path, self.model_name + '.pt')
        
        with open(json_file_path, "w") as outfile: 
            json.dump(data, outfile,indent=4)

        torch.save(self.dynoModel.state_dict(), pt_file_path)
        print(f"\nModel parameters saved to {modelPath}\n")

    def plot_training_loss(self):
        plt.figure()
        plt.plot(self.tLOSS, label = 'training')
        plt.plot(self.vLOSS, label = 'validation')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness Loss')
        plt.title("Model Training Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_evaluation(self,measured,estimated):
        plt.figure()
        plt.plot(estimated[self.output_variables[0]])
        loc,_ = plt.xticks()
        plt.close()
        # Plotting for each variable
        fig, axs = plt.subplots(len(self.output_variables), constrained_layout=True)

        for i, var in enumerate(self.output_variables):
            axs[i].set_title(var)
            axs[i].plot(measured[var], label='measured')
            axs[i].plot(estimated[var], label='estimated')
            axs[i].set_xticklabels(loc * self.dt, minor=False)
            axs[i].set_xlabel("Time (seconds)")
            axs[i].set_ylabel(var)

        plt.legend()
        plt.show()

    def plot_errors(self):
        # Plotting RMSE for each output variable
        x_values = np.linspace(1, self.num_datasets, self.num_datasets)

        # Plotting all in one figure
        plt.figure()
        plt.suptitle('RMSE for Output Variables')
        for i, var_name in enumerate(self.output_variables):
            plt.plot(x_values, self.case_rmse[:, i], label=f'{var_name}')
        plt.legend()
        plt.show()

        # Plotting in subplots
        fig, axs = plt.subplots(len(self.output_variables), constrained_layout=True)
        fig.suptitle("RMSE For Every Test Case")
        fig.supxlabel("Case Number")
        fig.supylabel("RMSE")

        colors = ['purple', 'blue', 'red', 'green', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']  # Extend as needed
        for i, var_name in enumerate(self.output_variables):
            axs[i].plot(x_values, self.case_rmse[:, i], label=f'{var_name}', color=colors[i % len(colors)])
            axs[i].legend()

        plt.show()

    def add_sensitivity_parameters(self, limit,weight,delta,inputs,outputs,sets):
        self.sens_limit = limit
        self.sens_weight = weight
        self.sens_delta = delta
        self.sens_inputs =  inputs
        self.sens_outputs = outputs
        self.sens_sets = sets

        self.sens_idx_set = {}
        for key in self.sens_sets:
            self.sens_idx_set[key] = np.where(self.input_variables == key)
            
        self.sens_bases = []
        vals = np.zeros(len(self.input_variables))
        keys = list(self.sens_sets)
        indices = np.zeros(len(keys), dtype = int)
        lens = np.zeros(len(keys), dtype = int)

        for i in range(len(keys)):
            lens[i] = len(self.sens_sets[keys[i]])
        
        self.sens_counter = 0

        self.add_sensitivity_bases(self.sens_bases,vals,keys,indices,lens,0)

        self.sens_kid = np.where(self.input_variables == self.sens_inputs[0])
        
        self.sens_kiq = np.where(self.input_variables == self.sens_inputs[1])

        self.sens_kod = np.where(self.output_variables == self.sens_outputs[0])
        self.sens_koq = np.where(self.output_variables == self.sens_outputs[1])

    def add_sensitivity_bases(self,bases,step_vals,keys,indices,lens,level):
        """Recursive function to add a set of operating points to the sensitivity evaluation set. Uses a depth-first approach. When the last channel number is processed, the recursion will back up to a previous channel number that was not fully processed yet. *Internal*

        Args:
            bases (list(float)[]): array of *step_vals* for operating points in the sensitivity evaluation set 
            step_vals (list(float)): input channel values for a model steady-state operating point 
            keys (list(str)): list of channel names from the sens_sets, each of these corresponds to a *level* of recursion
            indices (list(int)): keeps track of the channel number to resume processing whenever *level* reachs the last *key* 
            lens (list(int)): the number of operating point values for each named channel in *keys*
            level (int): enters with 0, backs up at the length of *keys* minus 1

        Yields:
            Appending to *bases*. Updates *sens_counter* in each call.

        Returns:
            None
        """

        self.sens_counter += 1

        key = keys[level]
        ary = self.sens_sets[key]
        idx = self.sens_idx_set[key]
        
        if level + 1 == len(keys): # add basecases at the lowest level
            for i in range(lens[level]):
                step_vals[idx] = ary[i]
                bases.append(step_vals.copy())
        
        else: # propagate this new value down to lower levels
            step_vals[idx] = ary[indices[level]]

        if level + 1 < len(keys):
            level += 1
            self.add_sensitivity_bases(bases, step_vals, keys, indices, lens, level)
        else:
            level -= 1
        while level >= 0:
            if indices[level]+1 >= lens[level]:
                level -= 1
            else:
                indices[level] += 1
                indices[level+1:] = 0
                self.add_sensitivity_bases(bases, step_vals, keys, indices, lens, level)

    def sensitivity_response(self,input,sens_matrix):

        # Normalize the input channels
        u_normalized = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.float64(input)),0),0)  # Clone to avoid modifying original tensors
        for j, key in enumerate(self.input_variables):
            mean = float(self.scaling_factors[key]['mean'])
            range_val = float(self.scaling_factors[key]['range'])
            u_normalized[0, :, j] = self.normalize(u_normalized[0, :, j], mean, range_val)
        
        u_normalized = u_normalized.to(torch.float32)
        # CREATE INPUT TENSOR

        y1 = self.dynoModel.F1(u_normalized)
        y2 = torch.unsqueeze(torch.sum(y1*sens_matrix,dim=2),0)
        y3 = self.dynoModel.F2(y2)

        # Convert model output to numpy array
        y3_denormalized = y3[0,:,:]
        
        # DENORMALIZE VALUES
        for i, key in enumerate(self.output_variables):
            y3_denormalized[:,i] = self.denormalize(y3_denormalized[:,i],float(self.scaling_factors[key]['mean']),float(self.scaling_factors[key]['range']))
            # i+=1

        d_sens = y3_denormalized[:,self.sens_kod]
        q_sens = y3_denormalized[:,self.sens_koq]

        return d_sens,q_sens


    def calculate_sensitivity(self):
        sens_max = torch.tensor(0.0)
        
        G1 = self.dynoModel.G1
        
        alpha1 = G1.alpha1
        alpha2 = G1.alpha2
        
        bc = G1.b_coeff
        a1 = 2*torch.tanh(alpha1)
        a1abs = torch.abs(a1)
        a2 = a1 + (2-a1abs)*torch.sigmoid(alpha2)-1
        ac = torch.cat((a1,a2), dim=2,)
        acbar = torch.cat((ac,torch.ones ((G1.b_coeff.shape[0], G1.b_coeff.shape[1], 1))),dim=2)

        sens_matrix = torch.divide(torch.sum(bc,dim=2),torch.sum(acbar,dim=2))
        sens_matrix = sens_matrix.to(torch.float32)
        
        for input in self.sens_bases:
            ud0 = input[self.sens_kid]
            uq0 = input[self.sens_kiq]
            
            ud1 = ud0 + self.sens_delta
            uq1 = uq0 + self.sens_delta
            
            yd0,yq0 = self.sensitivity_response(input,sens_matrix)

            input[self.sens_kid] = ud1
            input[self.sens_kiq] = uq0

            yd1,yq1 = self.sensitivity_response(input,sens_matrix)

            input[self.sens_kid] = ud0
            input[self.sens_kiq] = uq1
            
            yd2,yq2 = self.sensitivity_response(input,sens_matrix)

            input[self.sens_kiq] = uq0

            y_error = torch.stack([torch.abs(yd1-yd0),torch.abs(yq1-yq0),torch.abs(yd2-yd0),torch.abs(yq2-yq0)])

            sens = torch.max(y_error)/self.sens_delta

            sens_max = torch.max(sens_max,sens)
        
        sens_loss = self.sens_weight * sens_max
        
        return sens_loss

    def calculate_passivity_indices(self,):
        return 0
    
    def initialize_passivity_parameters(self,passivity_inputs,num_epochs):
        self.passivity_indices = []
        for input in passivity_inputs:
            self.passivity_indices.append(self.input_variables.tolist().index(input))
        
        self.passivity_num_epochs = num_epochs
        
        return 0
    
    def find_delta_epsilon(self,u,f_grad,y1,y4,rho):

        M1 = np.zeros((2,2))
        M1[0,0] = self.uy_dot(u,y1)+self.uy_dot(y1,u) - rho*self.uy_dot(y1,y1)
        M1[0,1] = self.uy_dot(u,y4)+self.uy_dot(y1,f_grad) - rho*self.uy_dot(y1,y4)
        M1[1,0] = self.uy_dot(f_grad,y1)+self.uy_dot(y4,u) - rho*self.uy_dot(y4,y1)
        M1[1,1] = self.uy_dot(f_grad,y4)+self.uy_dot(y4,f_grad) - rho*self.uy_dot(y4,y4)


        M2 = np.zeros((2,2))
        M2[0,0] = self.uy_dot(u,u)
        M2[0,1] = -self.uy_dot(u,f_grad)
        M2[1,0] = -self.uy_dot(f_grad,u)
        M2[1,1] = self.uy_dot(f_grad,f_grad)

        a = M1[1,1] * M2[0,1] - M1[0,1] * M2[1,1]
        b = M1[1,1] * M2[0,0] - M1[0,0] * M2[1,1]
        c = M1[0,1] * M2[0,0] - M1[0,0] * M2[0,1]

        numerator_1 = -b + np.sqrt(b**2 - 4 * a * c)
        numerator_2 = -b - np.sqrt(b**2 - 4 * a * c)
        denominator = 2 * a

        option_1 = numerator_1/denominator
        option_2 = numerator_2/denominator

        if option_1 > 0:
            if option_2 > 0 :
                delta = np.min([option_1,option_2])
            else:
                delta = option_1
        else:
            delta = option_2

        return delta
    
    def uy_dot(self,u,y):
        """
        Compute uᵀy or ‖u‖² if u and y are identical, with optional transposition
        if their first dimension equals N.

        Args:
            u (np.ndarray): 2D input array.
            y (np.ndarray): 2D input array.
            N (int): Reference dimension; if u.shape[0]==N, transpose u (and similarly y).

        Returns:
            float: If u and y (after any transpose) are identical, returns sum of squares of u;
                otherwise sum of columnwise inner products of u and y.
        """
        num_time_steps = len(u)
        
        u = np.asarray(u)
        y = np.asarray(y)

        # Transpose if first dimension matches N
        if u.shape[0] == num_time_steps:
            u = u.T
        if y.shape[0] == num_time_steps:
            y = y.T

        # If arrays are exactly equal, compute ∥u∥²
        if np.array_equal(u, y):
            return float(np.sum(u * u))

        # Otherwise compute sum of u[:,i]ᵀ y[:,i] over columns i
        # which is equivalent to sum of elementwise product
        return float(np.sum(u * y))

    def find_epsilon(self,u_initial,rho):
        u = u_initial
        
        num_time_steps = len(u[0,:,0].detach().numpy())
        
        time_values = np.linspace(0,self.dt*num_time_steps,num_time_steps)
        
        y1,_,_ = self.dynoModel(u,self.y0,self.u0)
        
        y1 = y1[0,:,:].detach().numpy().T

        u_act = []
        for index in self.passivity_indices:
            u_act.append(u[0,:,index].detach().numpy())
        u_act = np.array(u_act) 

        epsilon_history = []

        delta_history = []

        P = np.fliplr(np.eye(num_time_steps))

        for i in range(self.passivity_num_epochs):
            #f = (np.dot(u_act,y1) + np.dot(y1,u_act)) - rho * np.dot(y1,y1) / np.dot(u_act,u_act)
            f = (self.uy_dot(u_act,y1) + self.uy_dot(y1,u_act)) - rho * self.uy_dot(y1,y1) / self.uy_dot(u_act,u_act)

            u_in = u
            u_in[0,:,:] = torch.tensor(np.matmul(u[0,:,:].detach().numpy().T,P).T)

            y2,_,_ = self.dynoModel(u_in,self.y0,self.u0)
            y2 = np.matmul(y2[0,:,:].detach().numpy().T,P)
            u_temp = u
            j=0
            for index in self.passivity_indices:
                u_temp[0,:,index] = torch.tensor(y1[j,:])
                j = j+1

            u_temp[0,:,:] = torch.tensor(np.matmul(u_temp[0,:,:].detach().numpy().T,P).T)
            y3,_,_ = self.dynoModel(u_temp,self.y0,self.u0)
            y3 = np.matmul(y3[0,:,:].detach().numpy().T,P)
            #f_gradient = 2*(y1+y2-rho*y3 - (f*u_act.T))/ np.dot(u_act,u_act)
            f_gradient = 2 * (y1+y2-rho*y3 - (f*u_act)) / self.uy_dot(u_act,u_act)

            j=0
            u_temp = u
            for index in self.passivity_indices:
                u_temp[0,:,index] = torch.tensor(f_gradient[j,:])
                j = j+1

            y4,_,_ = self.dynoModel(u_temp,self.y0,self.u0)
            y4 = y4[0,:,:].detach().numpy()
            delta = self.find_delta_epsilon(u_act,f_gradient,y1,y4,rho)

            delta_history.append(delta)

            u_act = u_act - delta*f_gradient

            j=0
            for index in self.passivity_indices:
                u[0,:,index] = torch.tensor(u_act[j,:])
                j = j+1

            epsilon_history.append(f)

            y1,_,_ = self.dynoModel(u,self.y0,self.u0)
            
            y1 = y1[0,:,:].detach().numpy().T

        #epsilon_opt = (np.dot(u_act,y1) + np.dot(y1,u_act)) - rho * np.dot(y1,y1) / np.dot(u_act,u_act)
        epsilon_opt = (self.uy_dot(u_act,y1) + self.uy_dot(y1,u_act)) - rho * self.uy_dot(y1,y1) / self.uy_dot(u_act,u_act)

        epsilon_history.append(epsilon_opt)

        return epsilon_opt, u, epsilon_history, delta_history

    def find_passivity_type(self,u_initial):
        epsilon_opt,_,_,_ = self.find_epsilon(u_initial,rho=0)

        if epsilon_opt < 0:
            self.passivity_type = 'passivity_short'
        else:
            self.passivity_type = 'passive'

        print("The system is "+self.passivity_type+" with epsilon = "+str(epsilon_opt))

    def calculate_total_rmse_errors(self,bByCase=True):
        
        self.case_per_case_rmse = []
        out_size = len(self.output_variables)
        self.rmse_list = []
        self.total_rmse = np.zeros(out_size)
        self.total_mae = np.zeros(out_size)

        if bByCase:
            self.case_rmse = np.zeros([self.num_datasets, out_size])
            self.case_mae = np.zeros([self.num_datasets, out_size])
        else:
            self.case_rmse = None
            self.case_mae = None

        icase = 0
        i=0
        
        # Data Loader Loop
        for ub, y_true in self.total_dl: # batch loop

            # Simulate the model
            y_hat,_,_ = self.dynoModel(ub,self.y0,self.u0)
            
            # Assign the true solution to y1
            y1 = y_true.detach().numpy()

            # Assign the estimated solution to y2
            y2 = y_hat.detach().numpy()
            
            # Calculate the absolute error between y1 and y2
            y_err = np.abs(y1-y2)
            
            # Square the error
            y_sqr = y_err*y_err

            nb = y_err.shape[0]

            mae = np.mean (y_err, axis=1) # nb x ncol
            mse = np.mean(y_sqr, axis=1)
            rmse = np.sqrt(mse)
            self.case_per_case_rmse.append(rmse)
            
            self.total_mae += np.sum(mae, axis=0)
            self.total_rmse += np.sum(rmse, axis=0)

            if bByCase:
                iend = icase + nb
                self.case_mae[icase:iend,:] = mae[:,:]
                self.case_rmse[icase:iend,:] = mse[:,:]
                icase = iend
            i=i+1

        self.total_rmse = self.total_rmse / self.num_datasets
        self.total_mae /= self.num_datasets

        if bByCase:
            self.case_rmse = np.sqrt(self.case_rmse)
        
        self.plot_errors()

        print("Cumulative Sum of RMSE Errors: " + str(self.total_rmse))
        print("Mean of RMSE Errors: " + str(np.mean(self.case_per_case_rmse)))
        