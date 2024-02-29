# oneAPI_SA : FedML Project

Project: Federated Model Learning (FedML): Federated Deep Learning Models on the edge for enabling Continual Learning at the cloud.

Description: Deep learning (DL) models are usually deployed on cloud systems due to their large sizes and global accessibility.
With the demand of LLMs rapidly increasing for addressing latest issues, DL models cannot run inference with static parameters. Heterogeneous and latest data becomes key to keep the models updated with the latest parameters.
 
FedML operates on two key aspects of Deep Learning to address the above issues:

1) Federated Learning (FL) for obtaining edge heterogeneous data: We collaborate with various edge data generators to acquire the trends of the data. Since, the generated data is private to the cloud, applications can utilize FL to train their DL
 
2) Continuous Learning at the cloud: Global DL model running on the cloud will be updated with the Federated Edge Models, by this way the global server model keeps itself up to date for latest queries and information demand.

OneAPI Enabled Optimization and Inference: To optimize DL systems, we utilize rich libraries provided by Intel Extension for Pytorch. This enables us to employ Vector Neural Network Instruction set (VNNI) and Advanced Matrix Extensions (AMX) to accelerate training and inference on edge and cloud systems respectively.

Intel Extension For Pytorch: https://intel.github.io/intel-extension-for-pytorch/cpu/2.2.0+cpu/tutorials/introduction.html


---------------------------------------------------------------------------------------------------------------------------------------------------------------
System Definition (Beta)

Figure 1 in the Figures folder demonstrates the system model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Code Base:
A software bed has been created to simulate the described scenario. Please refer to the CodeBase folder to access these files.

The CodeBase consists of the following:
1) EdgeDevice.py : Python script defining edge device behaviour to enable federated learning.
2) ServerTest.py : Python script defining server hosting the large model.
3) main.py : Simulates the presented process.

Following Assumptions Have Been Made:
1) The server offloads the entire model to the edge device.
2) only 1 edge device is taken for demonstration.
3) The simulation completely runs on a single device, i.e communication protocols and the cloud is simulated.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Work in Progress:

1) Defining Communication stack for cloud and edge systems.
2) Moving Cloud Simulation to DevCloud (Running cloud routines on Intel Xeon Processors)
