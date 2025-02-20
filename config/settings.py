import torch 

NUM_CLASSES = 23
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPUDEVICE = torch.device("cpu")  # move to the CPU
NUM_EPOCHS = 100
