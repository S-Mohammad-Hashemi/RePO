# Enhancing Robustness Against Adversarial Examples in Network Intrusion Detection Systems

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

## Paper

**Abstract:**

The increase of cyber attacks in both the numbers and varieties in recent years demands to build a more sophisticated network intrusion detection system (NIDS). These NIDS perform better when they can monitor all the traffic traversing through the network like when being deployed on a Software-Defined Network (SDN). Because of the inability to detect zero-day attacks, signature-based NIDS which were traditionally used for detecting malicious traffic are beginning to get replaced by anomaly-based NIDS built on neural networks. However, recently it has been shown that such NIDS have their own drawback namely being vulnerable to the adversarial example attack. Moreover, they were mostly evaluated on the old datasets which don't represent the variety of attacks network systems might face these days. In this paper, we present Reconstruction from Partial Observation (RePO) as a new mechanism to build an NIDS with the help of denoising autoencoders capable of detecting different types of network attacks in a low false alert setting with an enhanced robustness against adversarial example attack. 
Our evaluation conducted on a dataset with a variety of network attacks shows denoising autoencoders can improve detection of malicious traffic by up to 29% in a normal setting and by up to 45% in an adversarial setting compared to other recently proposed anomaly detectors.

## Citation

```
@inproceedings{hashemi2020enhancing,
    title={Enhancing Robustness Against Adversarial Examples in Network Intrusion Detection Systems},
    author={Mohammad J. Hashemi and Eric Keller},
    booktitle={2020 IEEE Conference on Network Function Virtualization and Software Defined Networks (NFV-SDN)}, 
    year={2020},
}
```
