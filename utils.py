import torch

def create_device(force_cpu: bool):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device
