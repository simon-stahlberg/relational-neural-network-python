import torch

def create_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU: ", torch.cuda.get_device_name(0))
    # The MPS implementation does not yet support all operations that we use.
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("MPS is available. Using MPS.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device
