# Parking-Space-Inference
## Demo
![image](https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/fin.gif)
## Introduction
  This is an implementation of <br>
  "SENSOR BASED ON-STREET PARKING SPACE STATUS INFERENCE UPON A SEMI-SUPERVISED AND MULTI-TASK LEARNING NETWORK" <br>
  in TensorFlow for parking slots status inference, the goal of our method is dynamically determine the status of roadside parking spaces. 

  We propose well-designed deep learning networks for recognizing the sequential patterns of magnetic signals. The framework of the proposed system conposed of Coordinate Transform Module, Multi-task Module, and Temporal Module.
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/overview%20of%20%20model.png" width="60%">
### Coordinate Transform Module
To address the porblem which the coordinate of installed sensors is unknown.
### Multi-task Module
To take advantage the information from both labeled and unlabeled data and hence strengthen the training model’s resistance to external noises and its adaptability to aforementioned signal variations. 
### Temporal Module
Use LSTM[11] to grasp the logic characteristics of the above state transition and observe them for a long time.
## Challenges
In real testing environment, the signal responses of magnetic sensors are noisy and unstable which makes it difficult to determine the parking status robustly. These possible challenges are summarized as follows:<br>
(1) The interruption from environment magnetic fields and environment noise.[1]<br>
(2) The variety of magnetic signals due to vehicle types.[4]<br>
(3) The interruption by moving vehicles.[2]<br>
(4) The non-unified coordination of magnetic sensors.[3]<br>
(5) The annoying magnetic responses caused by the status changing of neighboring spaces.[3]<br>


## Dataset:
### Data format
Source data contains total of 62,084 data points.<br>
Target data contains total of 16,444 data points.<br>
We obtain total 1,552 source sequences and 411 target sequences by utilize shift windows with size 80, and stride with size 40.<br>
(1)Example of collected data
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/collected%20.png" width="60%">
<br>
(2)Example of disturbance data
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/disturbance%20.png" width="60%">
## Training:
	-

## Experiment:
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/c.png" width="40%">
<br>
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/s.png" width="40%">
<br>
<img src="https://github.com/Lilyo/Parking-Space-Inference/blob/master/fig/a.png" width="40%">
## Evaluation:
	-

## You should have this tree structure in the current folder:

~/USER/ParkingSpaceInference$<br>
.<br>
├── train.py.py<br>
├── tools.py<br>
├── inference.py<br>
├── help_function.py<br>
├── load_data.py<br>
├── next_batch.py<br>
├── stn1d.py<br>
├── ID_0_C2_AllLabel.txt<br>
└── ID_0_Angle_0_Disturbance_AllLabel_v1.txt<br>



## Citation
"Sensor Based On-street Parking Space Status Inference Upon A Semi-supervised And Multi-task Learning Network", Computer Vision, Graphic and Image Processing (CVGIP), Aug., 2, 2018<br>
    @article{ParkingSpaceInference,<br>
        title={Sensor Based On-street Parking Space Status Inference Upon A Semi-supervised And Multi-task Learning Network},<br>
        author={You-Feng Wu, Hoang Tran Vu, Ching-Chun Huang},<br>
        year={2018}<br>
    }

## Reference
[1] Z. Zhang, X. Li, H. Yuan, and F. Yu, “A Street Parking System Using Wireless Sensor Networks,” Int. J. Distributed Sensor Networks, vol. 7, no. 2, pp. 153-163, 2013.<br>
[2] H. Zhu and F. Yu, “A cross-correlation technique for vehicle detections in wireless magnetic sensor network,” IEEE Sensors J., vol. 16, no. 11, pp. 4484-4494, Jun. 2016.<br>
[3] Z. He, H. Zhu, and F. Yu, “A vehicle detection algorithm based on wireless magnetic sensor networks,” in IEEE Int. Conf. on Information Science and Technology, Shenzhen, China, 2014, pp. 727-730.<br>
[4] H. Zhu and F. Yu, “A vehicle parking detection method based on correlation of magnetic signals,” International Journal of Distributed Sensor Networks, vol. 2015, pp. 1-13, 2015.<br>
[11] S. Hochreiter and J Schmidhuber, “Long short-term memory,” Neural Computation, pp. 1735–1780, 1997.
