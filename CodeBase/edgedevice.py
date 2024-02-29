import torch 

import intel_extension_for_pytorch as ipex

import torch.nn as nn 

import torch.optim as optim

class EdgeDevice:

    def __init__(self):

        self.DeviceID = -1 

    def SetDeviceID(self, id):

        self.DeviceID = id

        return None 

    def ReceiveModelfromServer(self, model : nn.Module):

        TrainModel = model

        print("--------- Device: Succesfully Received the Model from Server ---------")
        
        print(" ")

        return TrainModel 

    def SendModeltoServer(self, model : nn.Module):

        return (self.DeviceID, model)

    def FxLearn(self, model : nn.Module, trainloader):

        print(f"----------------- Initiating Device Training on Device {self.DeviceID}  ----------------")

        print(" ")

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9)
        
        model.train()

        model, optimizer = ipex.optimize(model, optimizer = optimizer)

        epochs = 20

        for epoch in range(epochs):

            for i, (images, labels) in enumerate(trainloader):

                images = images.reshape(-1, 28*28)

                outs = model(images)

                loss = criterion(outs, labels)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
            
            print(f"------------Epoch Completed = {epoch} || Loss = {loss.item()} ------------")
            
        print(" ")

        print("Device: Training Completed Successfully on the Device")
        
        return model

    












        










