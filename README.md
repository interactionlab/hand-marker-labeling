# Real-time Hand Marker Labeling System using Deep Learning
This repository contains a deep learning based approach to label hand-markers in real-time. 
All preprocessing steps, the training process, the trained model itself, and a unity plugin to receive and visualize the labeled data are provided in this repository.

## Content
We provide [jupyter notbooks](Training) to create a machine learning model from the recorded motion capture data step-by-step.
Therefore the data has to be exported to .csv files.

For real-time labeling, the motion capture data can be streamed to the [Labeling Client](Labeling%20Client).
It uses a NatNet client to receive the data and runs the labelNetwork to get the hand labels.

The [Unity Client](Unity%20Client) can receive the labeled data and fires a HandDataEvent whenever data is received.
To use the Unity Client in you own project import the [HandTrackingClient.unitypackage](Unity%20Client/HandTrackingClient.unitypackage) and add the HandTrackingClient to your scene.

We modified the NatNet SDK's python client to be able to receive labeled and unlabeled markers in the Labeling Client.
