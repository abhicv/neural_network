from ctypes import *
from encodings import utf_8
import os

class NEURON(Structure):
    _fields_ = [("activation", c_float),
                ("z", c_float),
                ("weights", POINTER(c_float)),
                ("bias", c_float),
                ("dcost_da", c_float)]

class LAYER(Structure):
    _fields_ = [("neurons", POINTER(NEURON)),
                ("neuron_count", c_uint32)]

class NETWORK(Structure):
    _fields_ = [("layers", POINTER(LAYER)),
                ("layer_count", c_uint32),
                ("topology", POINTER(c_uint32)),
                ("learn_rate", c_float)]

def create_network(topology, learn_rate):
    return lib.CreateNetwork((c_uint32 * len(topology))(*topology), len(topology), learn_rate)

def feedforward(network, input):
    lib.FeedForward(pointer(network), (c_float * len(input))(*(input)), len(input))

def backpropagate(network, target):
    lib.BackPropagate(pointer(network), (c_float * len(target))(*(target)), len(target))

def load_network_from_file(file_name):
    return lib.LoadNetworkFromFile(file_name.encode('utf_8'))

lib = CDLL(os.getcwd() + "/libnn.so")

lib.CreateNetwork.argtypes = [POINTER(c_uint32), c_uint32, c_float]
lib.CreateNetwork.restype = NETWORK

lib.FeedForward.argtypes = [POINTER(NETWORK), POINTER(c_float), c_uint32]

lib.BackPropagate.argtypes = [POINTER(NETWORK), POINTER(c_float), c_uint32]

lib.ComputeCost.argtypes = [NETWORK, POINTER(c_float), c_uint32]
lib.ComputeCost.restype = c_float

lib.LoadNetworkFromFile.argtypes = [c_char_p]
lib.LoadNetworkFromFile.restype = NETWORK
