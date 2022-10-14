# Quantization using Microsoft NNI
### Authors
Jinam Shah </br>
Saumil Shah

## Folder structure

The python file **model_VAE** contains the main definition of the model i.e. VAE. This is inspired from the pytorch/examples github but is a little different in the actual definition of the model layers.

The **quantization** contains the methods used for model quanitization and the different configurations used for model it. It uses three quantizers and two different configurations from NNI.

The log_files folder contains the log files for the different configurations tried.
```
File naming format: <Quantizer used>_<Config used>_<Device used>.txt
```
We have also consolidated the NNI configurations into `configuration_list.txt`

## Code execution
The pruning example can be run by simply triggering `python3 quantization_VAE.py`

Lines 11 and 12 of the **quantization_VAE** file provide the choice between the different pruners and the different configurations for pruning on the model defined in **model_VAE**.
```
quantization_used = "LsqQuantizer" or "NaiveQuantizer" or "QAT_Quantizer"
config_choice = "config_list_1" or "config_list_2"

Config list details for QAE Quantizer and LSQ Quantizer:
Config list 1
configuration_list = [{
                'quant_types': ['input', 'weight'],
                'quant_bits': {'input': 8, 'weight': 8},
                'op_types': ['Linear']
            },
            {
                 'quant_types': ['output'],
                 'quant_bits': {'output': 8},
                 'op_types': ['Linear']
             }]
Config list 2
configuration_list = [{
                 'quant_types': ['input', 'weight'],
                 'quant_bits': {'input': 8, 'weight': 8},
                 'op_names': ['fc1', 'fc4']
             }]

Config list details for Naive Quantizer
Config list 1
configuration_list = [{
                 'quant_types': ['weight'],
                 'quant_bits': {'weight': 8},
                ' op_types': ['Linear']
             }]
Config list 2
configuration_list = [{
                 'quant_types': ['weight'],
                 'quant_bits': {'weight': 8},
                 'op_names': ['fc1', 'fc4']
             }]
```

We also provide the option of using CPU or GPU of the machine that you are running the code on. Although for this, a code change in the model definition (model_VAE.py) is required. As a simple way of doing this, we just directly change the argument defined for `no-cuda` (line 15).
For using GPU:
```
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
```
For using CPU
```
parser.add_argument('--no-cuda', action='store_true', default=True,help='disables CUDA training')
```
