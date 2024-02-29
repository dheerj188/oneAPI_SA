# This code demonstrates the function of a server hosting deep learning workload

# Server Task: 
# 1) provide continous inference for incoming services 
# 2) Learn incoming parameters from federated edge cores

# import libraries 

import multiprocessing 

import torch 

import intel_extension_for_pytorch as ipex

import torch.nn as nn 

import torch.optim as optim 

# define the global model (LLMs/Large scale vision models)

class ServerModel(nn.Module):

    def __init__(self):

        pass

    def forward(X):

        pass

# For this version, we will be training a simple model. 

# definition of server class 

class Server:

    def __init__(self, EdgeDevices, model):

        # Global Server Parameters 

        # These are server parameters 

        self.InferenceModel = model

        self.TrainModel = model

        self.LayerMap = {}

        # cloud-edge database. 

        self.EdgeDevices = EdgeDevices

        self.FxModels = [model for i in range(self.EdgeDevices)]

        # Dedicated edge core parameters

        self.FxLayerMaps = []

    # Helper functions
    # Helper functions include initializers, model optimizers

    def InitializeFxLayers(self):

        for id in range(self.EdgeDevices):

            LID = 0

            LayerMap = {}

            for _ , layer in self.TrainModel.named_parameters():

                LayerMap[LID] = layer

                LID += 1
            
            self.FxLayerMaps.append(LayerMap)

        return None 

    def InitializeServerModel(self):

        LID = 0

        for _ , layer in self.TrainModel.named_parameters():

            self.LayerMap[LID] = layer

            LID += 1 

        return None

    # Optimize the model for inference 

    # Routines for optimizing and running inference 

    def OptimizeInferenceModel(self):

        self.InferenceModel.eval()

        self.InferenceModel = ipex.optimize(self.InferenceModel)

        return None 

    def RunInference(self, data):

        with torch.no_grad():

            response = model(data)

        return response 

    # Update the weights with new parameters

    def AggregateFxParameters(self):

        return None 

    def UpdateTrainModel(self):

        return None 

    def UpdateInferenceModel(self):

        self.InferenceModel = self.FxTrainModel

        return None 










