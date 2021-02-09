# Visibility_Correction_Pytorch

Model to correct visibility artefact in Pytorch using FPN, ASPP, and pretrained Resunet, to optimize <br/>
Modified from: https://github.com/ellisdg/3DUnetCNN and https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter <br/>
<br/>
User guide:

* Create the trainingset and testset folder as tr and ts
* Create a model and results folder to store trained models and reconstructed model.
* Modify the parameters of the model using the config file
* Run main to train the model
* Run predict to reconstruct images from testset
* Code currently working for matlab data, modify data_prepare to change data type
