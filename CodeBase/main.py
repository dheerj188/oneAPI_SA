import edgedevice as ed

import ServerTest as server

import torchvision

import torchvision.transforms as transforms 

from torch.utils.data import DataLoader

"""Simulation Begins Here"""

# ------------------------------ Server Initialize Space ------------------------------#

# defining input and output shape of cloud model 

input_shape = 784

output_shape = 10

model = server.NeuralNet(input_shape, output_shape)

num_edge_devices = 1

server = server.Server(num_edge_devices, model)

device = ed.EdgeDevice()

devices = [device]

server.InitializeFxLayers()

server.InitializeAggInfLayers()

server.AssignDeviceIDs(devices)

# -------------------- Communication Between Cloud and Edge device -----------

print(" ------------------------ Initiating Communicaion Between Cloud and Devices ----------------------")

print(" ")

model_server = server.SendModeltoDevice(device.DeviceID)

model_device = device.ReceiveModelfromServer(model_server)

# ------------------------- Device Phase ---------------------------------------

# Loading and Data preprocessing done here 

# dataset hyperparameter definition 

input_size = 784

batch_size = 100

num_classes = 10

# importing MNIST dataset and setting up the dataloader 

train_dataset = torchvision.datasets.MNIST('./data', train = True, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

# Federated Learning on Device

device_model = device.FxLearn(model_device, train_loader)

# --------------------------- Communication Between Edge Device and Cloud ------------------------ 

device_id, device_trained_model = device.SendModeltoServer(device_model)

server.RecieveModelfromDevice(device_trained_model, device_id, optimize = False)

# ---------------------------- Server Model Update Phase -------------------------

server.AggregateFxModels()

server.UpdateInferenceModel(alpha = 0.2)

# ------------------------- Server Run Inference ---------------------------------------

server.OptimizeInferenceModel()

print(" ")

print("Inference Service is Online")

print(" ")

for tick in range(100):

    user_data = server.ReceiveDatafromUser()

    response = server.RunInference(user_data)

print("Process Completed Succesfully")














