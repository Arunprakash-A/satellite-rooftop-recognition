import torch
from matplotlib import pyplot as plt

from model import CnnModel
import utils

convNet = CnnModel()
convNet.load_state_dict(torch.load('./chkpoints/parameters'))

np = convNet.named_parameters()
