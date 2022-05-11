# AutoEncoderHeartCT
An auto-encoder for the heart 2.5-D CT images 

Autoencoders are deep machine learning models that can be used as a dimention reduction technique to condense a large amount of information into a small feature space for classification purposes. In this repository, you can find a trained model as well as the code to train a model from scratch to make an autoencoder that reconstructs 2.5 dimentional data. The data is obtained from COVID-19 patients heart via 3-D X-ray computed tomography. Each of the input images include 10 equally-distanced slices from base to apex of the heart. 

Training and test data are stored and read via HDF5 format stored in a field named 'X' with 8-bit unsigned integers fetween 0 and 255. 

