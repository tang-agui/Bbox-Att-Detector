# Bbox_Att_Detector


## Introduction
This paper propose a novel framework for black-box adversarial patch attacks against aerial imagery object detectors using differential evolution (DE). Specifically, we first propose a dimensionality reduction strategy to address the dimensionality curse in high-dimensional optimization problems and improve optimization efficiency. Then, we design three universal fitness functions to help DE find promising solutions within a limited computational budget according to the diverse outputs of the detectors. Finally, we conduct extensive experiments on the DOTA dataset against state-of-the-art object detectors such as YOLOv3, YOLOv4, and Faster R-CNN. Results show that our method exhibits superior performance in addressing black-box attacks on aerial imagery object detection. To the best of our knowledge, this is the first work to explore the use of DE in black-box adversarial patch attacks against aerial imagery object detectors.

## Rerquiremens
Our code was built on [Pytorch](https://pytorch.org/).

## Usage
Run
`python TFNS.py.`

You may need to replace the imgdir, cfg and weightfile in `TFNS.py`  with yours. 


## References
- [DOTA](https://captain-whu.github.io/DOTA/code.html)

