# sit-to-stand-phase-identification-using-wearable-inertial-sensors
The core code is corresponding to the following paper:
Title:A Novel CNN-BiLSTM ensemble model with attention mechanism for sit-to-stand phase identification using wearable inertial sensors
Authors:Xin Chen, Shibo Cai*，Longjie Yu, Xiaoling Li, Bingfei Fan, Mingyu Du, Tao Liu, and Guanjun Bao

##Equipmental settings
The following equipment was utilized for the experimental settings(See the second section of the EXPERIMENTAL VALIDATION for details):
- Two AMTI force plates
- A set of Optima signal amplifiers
- Two wireless inertial sensors
- A Awinda Station receiver

##Computing Environment
OS: Microsoft Windows 10
CPU:Intel Core  i7-13700K
Memory:32 GB RAM
GPU:NVIDIA GeForce RTX 3090

##Software and Libraries
- IMU data processing、phase segmentaion and build STS-PD dataset(See "IMU dataprocessing and phase segmentaion and build dataset" folder for specific running code）: MATLAB
- Machine learning and Nerual network algorithms Development Environment:PyCharm
Python:3.7
CUDA:11.7
Keras:2.10
NumPy:1.21.5
Scikit-Learn:1.0.2

##Sit-to-Stand Transition Phases
The sit-to-stand transition was divided into five  phases:
1.Initial Sitting Phase
2.Flexion Momentum Phase
3.Momentum Transfer Phase
4. Extension Phase
5.Stable Standing Phase

 ##Phase Identification Algorithms
For the accurate identification of transition phases, the following methods were employed:
1.Threshold Method
2.CNN-BiLSTM-Attention Algorithm

The study involved a comparative analysis between Machine learning algorithms and Nerual network algorithms. The following algorithms were compared:
##Machine Learning Algorithms
See "Machine learing algorithm" folder for specific running code
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- 1-Nearest Neighbor (1NN)
- Decision Trees (DT)
- Logistic Regression (LR)
- Random Forest (RF)

##Nerual network Algorithm
See "Nerual network algorithm and Gated transformer" folder for specific running code
- MLP
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- CNN-Bi-directional LSTM (CNN-Bi-LSTM)
- Bi-directional LSTM (Bi-LSTM)
- Gated Transformer
- CNN-BiLSTM-Attention
