# Import the necessary modules for inferencing on server 

import torch 

import intel_extension_for_pytorch as ipex

import torch.nn as nn 

import torch.optim as optim 

import random 

"""Define the Neural Network Architecture"""

class NeuralNet(nn.Module):
    
    def __init__(self, input_size, output_size):
        
        super(NeuralNet, self).__init__()
        
        self.InputLayer = nn.Linear(input_size, 100)
        
        self.relu = nn.ReLU()
        
        self.hl = nn.Linear(100, 100)
        
        self.OutputLayer = nn.Linear(100, output_size)
        
    def forward(self, x):
        
        response = self.InputLayer(x)
        
        response = self.relu(response)
        
        response = self.hl(response)
        
        response = self.relu(response)
        
        response = self.OutputLayer(response)
        
        return response

class Server: 

    def __init__(self, EdgeDevices, model):

        self.InferenceModel = model 

        self.TrainModel = model 

        self.EdgeDevices = EdgeDevices 

        self.DeviceIDs = [id for id in range(EdgeDevices)]

        self.FxModels = [model for i in range(self.EdgeDevices)]

        self.AggregateModel = model 

        self.FxLayerMaps = []

        self.FxLID = 0

        self.AgLayerMap = {}

        self.AgLID = 0 

        self.InfLayerMap = {}

        self.InfLID = 0

    def InitializeFxLayers(self):

        for id in range(self.EdgeDevices):

            LID = 0

            LayerMap = {}

            for _ , layer in self.TrainModel.named_parameters():

                LayerMap[LID] = layer

                LID += 1
            
            self.FxLayerMaps.append(LayerMap)

        self.FxLID = LID

        print("--------- Server: Edge Trained Layers Initialized on Server Database ---------")

        print(" ")

        return None 

    def InitializeAggInfLayers(self):

        LID = 0

        for _ , layer in self.AggregateModel.named_parameters():

            self.AgLayerMap[LID] = layer

            self.InfLayerMap[LID] = layer

            LID += 1

        self.AgLID = LID

        self.InfLID = LID

        print("--------- Server: Aggregate Model Layers Initialized in Server Database --------- ")

        print(" ")

        return None 

    def OptimizeInferenceModel(self):

        self.InferenceModel.eval()

        self.InferenceModel = ipex.optimize(self.InferenceModel)

        return None 

    def AssignDeviceIDs(self, Devices):

        for id in range(self.EdgeDevices):

            Devices[id].SetDeviceID(self.DeviceIDs[id])

        print("--------- Server: IDs Assigned for Associated Devices ---------")

        print(" ")
        
        return None 

    def SendModeltoDevice(self, device_id):

        model = self.FxModels[device_id]

        return model 

    def RecieveModelfromDevice(self, model, device_id, optimize: bool):

        if (optimize):

            model = ipex.optimize(model)

        self.FxModels[device_id] = model

        print(" ------------------- Server: Successfully Received Trained Model from Device ---------------------- ")

        print(" ")
        return None 

    def ReceiveDatafromUser(self):

        Data = torch.tensor([[random.random() for x in range(28)] for y in range(28)], dtype = torch.float32)

        return Data  

    def RunInference(self, data):

        data = data.reshape(-1, 28*28)

        with torch.no_grad():

            response = self.InferenceModel(data)

        return response

    def AggregateFxModels(self):

        for mo, device_id in zip(self.FxModels, self.DeviceIDs):

            for lid in range(self.FxLID):

                with torch.no_grad():

                    self.AgLayerMap[lid] += self.FxLayerMaps[device_id].get(lid)

        for lid in range(self.AgLID):

            with torch.no_grad():

                self.AgLayerMap[lid] = (1/(self.EdgeDevices + 1)) * (self.AgLayerMap.get(lid))

        return None 

    def UpdateInferenceModel(self, alpha):

        for lid in range(self.InfLID):

            with torch.no_grad():

                self.InfLayerMap[lid] = (1 - alpha) * (self.InfLayerMap.get(lid)) + (alpha) * (self.AgLayerMap.get(lid))

        return None 

      






            


        
            

            



        

            



            



        



        

            




    
















