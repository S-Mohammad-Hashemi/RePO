Enhancing Robustness Against Adversarial Examples in Network Intrusion Detection Systems

This repository contains the code to train and test a network intrusion detection system (NIDS) with the Reconstruction from Partial Observation (RePO) technique as described in our paper. The NIDS can be trained on low-level features extracted from packet headers and also high-level features extracted from a whole flow.

Before running the codes, the datasets should be first extracted.
```
./extract_packet_data.sh
./extract_flow_data.sh
```

Flow-level features are directly used from CICIDS2017 dataset (https://www.unb.ca/cic/datasets/ids-2017.html). Packet-level features are extracted from PCAP files as described in our paper.

Dependencies:
```
Python 3.7.7
TensorFlow 2.1.0
Numpy 1.18.1
Pandas 1.0.3
```

The models we trained are available in *models* directory. By going through *Packet_based_RePO_Normal_Test* and *Flow_based_RePO_Normal_Test* notebooks our results in a normal setting can be reproduced. For reproducing the results in an adversarial setting *Packet_based_RePO_Adversarial* and *Flow_based_RePO_Adversarial* notebooks are provided. In these two notebooks, the *attack_type* variable should be changed to get the results for each network attack separately.
