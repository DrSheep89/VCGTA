# VCGTA: Viewport-aware Coalition Game Theory Adaptive Streaming for Live 360$^\circ$ Videos



## Abstract

As the predominating internet application, live 360$^\circ$ video streaming faces challenges of high video quality and limited network bandwidth. Dynamic Adaptive Streaming over HTTP (DASH) and tiling techniques make the delivery of live 360$^\circ$ streaming more flexible. However, these fine-grained approaches may bring additional computational overhead and make bitrate allocation more complex. As live 360$^\circ$ streaming is susceptible to delays, lightweight adaptive streaming can be essential for delivery. In this work, we propose _VCGTA_, a lightweight adaptive bitrate approach that can significantly improve the user's quality of experience (QoE) and transmission efficiency for live 360$^\circ$ streaming. _VCGTA_ can learn the user's viewing behavior by predicting the user's future viewport in real time so that the viewport area can be prioritized. Moreover, we innovatively model the tile-level bitrate allocation problem as a coalition game. Simulating the competitive behavior of tiles, _VCGTA_ is capable of improving robustness under complex network conditions. In addition to certain QoE improvements, _VCGTA_ can also significantly reduce computational complexity. A series of experiments based on the public dataset show that _VCGTA_ outperforms the existing adaptive bitrate algorithms in terms of various QoE models and bandwidth consumption.



## The Overview of VCGTA

![overview](https://s2.loli.net/2022/04/13/KRiMlu2QjZH6npU.png)





## Environment

- Windows 10
- PyTorch 1.8.1
- Python 3.7
- OpenCV and so on



#### Code Guide

- [Arguments.py](Arguments.py) : model hyperparameter settings
- [ConvLSTM.py](ConvLSTM.py) ：Convolutional LSTM cell and network
- [load_data.py](load_data.py) ：read relevant data from files

- [FoV.py](FoV.py) ：main code for online viewport prediction
- [get_fov.py](get_fov.py) ：get results from viewport prediction
- [Net_sim.py](Net_sim.py) ：simulation environment
- [tileSet.py](tileSet.py) ：store tile information

- [VCGTA.py](VCGTA.py) ：main ABR logic of VCGTA
- [abr_sim.py](abr_sim.py) ：main simulation function  of ABR algorithm

:warning: Please be careful to adjust the path settings in [Arguments.py](Arguments.py) and other script files before using the relevant code.

