from ctypes import *
from encodings import utf_8
import os

class VALUE(Structure):
    _fields_ = [("data", c_float),
                ("grad", c_float)]

class NEURON(Structure):
    _fields_ = [("activation", c_float),
                ("z", c_float),
                ("weights", POINTER(VALUE)),
                ("bias", VALUE),
                ("dcost_da", c_float),
                ("dead", c_bool)]

class LAYER(Structure):
    _fields_ = [("neurons", POINTER(NEURON)),
                ("neuron_count", c_uint32),
                ("activation_func", c_uint32),
                ("drop_rate", c_float)]

class NETWORK(Structure):
    _fields_ = [("layers", POINTER(LAYER)),
                ("layer_count", c_uint32),
                ("learn_rate", c_float)]

# lib = CDLL(os.getcwd() + "/libnn.so")
lib = WinDLL(os.getcwd() + "/libnn.dll")

print(lib)

lib.FeedForward.argtypes = [POINTER(NETWORK), POINTER(c_float), c_uint32]
print(lib.FeedForward)

lib.LoadNetworkFromFile.argtypes = [c_char_p]
lib.LoadNetworkFromFile.restype = NETWORK
print(lib.LoadNetworkFromFile)

def feedforward(network, input):
    lib.FeedForward(pointer(network), (c_float * len(input))(*(input)), len(input))

def load_network_from_file(file_name):
    return lib.LoadNetworkFromFile(file_name.encode('utf_8'))
