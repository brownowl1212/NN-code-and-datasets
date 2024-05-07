This repository houses the source material for Nirosan Pragash's Master's Dissertation - Data-Driven Estimation of Battery State of Health (SOH) and Remaining Useful Life (RUL). It encompasses both the code and datasets utilised within this project.

The central component of this model entails the development of a neural network (NN) capable of predicting the entire charge curve of a lithium-ion battery from a limited section of said curve. Subsequently, this prediction aids in determining the maximum capacity and thereby the state of health of the battery. This process involved dataset optimisation, and NN tuning, training and testing. 

Components within the repository include:
Data Processing Code: Housing the scripts used for processing raw battery degradation datasets.

Datasets: Comprising optimised datasets intended for use by the neural network, available both in CSV and text formats.

Single Dataset NN Code: Consisting of code designed to train and evaluate a NN model on individual battery degradation datasets.

Multi Dataset NN Code: Encompassing code tailored for training and evaluating a NN model on multiple battery degradation datasets, ensuring accuracy across all datasets.
