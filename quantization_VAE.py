from model_VAE import VAE, device, trainer, evaluator
from torch.optim import Adam
from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer, QAT_Quantizer, LsqQuantizer
# from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
# import torch_tensorrt
import torch
import time
import os

epochs = 15
quantization_used = "LsqQuantizer"  # -> (NaiveQuantizer, QAT_Quantizer, LsqQuantizer)
config_choice = "config_list_2"  # -> (config_list_1, config_list_2)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':

    print("\nDEVICE BEING USED: ", device, "\n")

    # Defined original unpruned model
    model = VAE().to(device)

    print("ORIGINAL UN-QUANTIZED MODEL: \n\n", model, "\n\n")

    # Starting time for unpruned model
    start_time = time.time()

    # Running the pre-training stage with original unpruned model
    optimizer = Adam(model.parameters(), lr=1e-2)
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)

    # Ending time for unpruned model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF UN-QUANTIZED MODEL: ", exec_time, "SECONDS", "\n\n")

    # Specifying two sets of configuration
    if quantization_used != "NaiveQuantizer":

        if config_choice == "config_list_1":
            # Defining the configuration list for pruning
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

        else:
            # Defining the alternate configuration list for pruning
            configuration_list = [{
                'quant_types': ['input', 'weight'],
                'quant_bits': {'input': 8, 'weight': 8},
                'op_names': ['fc1', 'fc4']
            }]

    else:
        if config_choice == "config_list_1":
            # Defining the configuration list for pruning
            configuration_list = [{
                'quant_types': ['weight'],
                'quant_bits': {'weight': 8},
                'op_types': ['Linear']
            }]

        else:
            # Defining the alternate configuration list for pruning
            configuration_list = [{
                'quant_types': ['weight'],
                'quant_bits': {'weight': 8},
                'op_names': ['fc1', 'fc4']
            }]

    # Defining the pruner to be used
    if quantization_used == "NaiveQuantizer":
        # Wrapping the original model with quantization wrapper
        quantizer = NaiveQuantizer(model, configuration_list, optimizer)

    elif quantization_used == "QAT_Quantizer":
        # Wrapping the original model with quantization wrapper
        dummy_input = torch.rand(32, 1, 28, 28).to(device)
        quantizer = QAT_Quantizer(model, configuration_list, optimizer, dummy_input)

    elif quantization_used == "LsqQuantizer":
        # Wrapping the original model with quantization wrapper
        dummy_input = torch.rand(32, 1, 28, 28).to(device)
        quantizer = LsqQuantizer(model, configuration_list, optimizer, dummy_input)

    print("QUANTIZER WRAPPED MODEL WITH {}: \n\n".format(quantization_used), model, "\n\n")

    # Next, compressing the model and generating masks
    quantizer.compress()

    # Starting time for pruned model
    start_time = time.time()

    optimizer = Adam(model.parameters(), lr=1e-2)
    # Running the pre-training stage with pruned model3)
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)

    # Ending time for pruned model
    end_time = time.time()

    # The total execution time of pruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF QUANTIZED MODEL: ", exec_time, "SECONDS", "\n\n")

    if quantization_used != "NaiveQuantizer":
        model_path = "./config_log/vae_model.pth"
        calibration_path = "./config_log/vae_calibration.pth"
        calibration_config = quantizer.export_model(model_path, calibration_path)
        print("calibration_config: ", calibration_config)