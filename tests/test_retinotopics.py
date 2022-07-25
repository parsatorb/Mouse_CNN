import mousenet
import torch
import pdb

def test_retinotopics_loads():
    model = mousenet.load(architecture="retinotopic", pretraining=None)
    input = torch.rand(1, 3, 64, 64)
    results = model(input)
    print(results.shape)
    pdb.set_trace()
    return true

def test_retinotopics_loads_in_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mousenet.load(architecture="default", pretraining=None)
    model.to(device)
    input = torch.rand(1, 3, 64, 64).to(device)
    results = model(input)
    print(results.shape)
    pdb.set_trace()