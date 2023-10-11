# Texture Classification with T-CNN

# Description
Training and classifying DTD textures from scratch using the T-CNN architecture, and compares the results with LeNet5.
- Dataset used: https://paperswithcode.com/dataset/dtd
- T-CNN network architecture paper: https://ieeexplore.ieee.org/document/8237882
- Uses filter banks which improves the performance of CNNs while greatly reducing the memory usage and computation.
- Achieved the stated accuracy of 0.28 ± 0.01 in the paper which outperforms LeNet-5 by 78.57%.


# Abstract
Deep learning has been demonstrated to achieve excellent results for image classification and object detection. However, the impact of deep learning on video analysis has been limited due to complexity of video data and lack of annotations. Previous convolutional neural networks (CNN) based video action detection approaches usually consist of two major steps: frame-level action proposal generation and association of proposals across frames. Also, most of these methods employ two-stream CNN framework to handle spatial and temporal feature separately. In this paper, we propose an end-to-end deep network called Tube Convolutional Neural Network (T-CNN) for action detection in videos. The proposed architecture is a unified deep network that is able to recognize and localize action based on 3D convolution features. A video is first divided into equal length clips and next for each clip a set of tube proposals are generated based on 3D Convolutional Network (ConvNet) features. Finally, the tube proposals of different clips are linked together employing network flow and spatio-temporal action detection is performed using these linked video proposals. Extensive experiments on several video datasets demonstrate the superior performance of T-CNN for classifying and localizing actions in both trimmed and untrimmed videos compared to state-of-the-arts.
