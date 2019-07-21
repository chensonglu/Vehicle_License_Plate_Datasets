# Simultaneous End-to-End Vehicle and License Plate Detection with Multi-Branch Attention Neural Network ***Supplementary Materials***

## III. METHODOLOGY
### A. Base Network
The details of the backbone network are shown in Table I without the ReLU activation function, where layers with **bold** font denote the transformed VGG-16 and the remainders represent the extra layers. As for parameters, "k" means kernel size, "s" stride size, and "d" dilation[1] parameters. In addition, "ceil" means rounding up the size if not divisible, so the size of conv3_3 maybe not exactly 2 times of conv4_1. In addition, layers marked with "\*" are candidates for detection head layers.

![backbone](extras/backbone.png)

### B. Detection Branch
The scale of features in different layers may be quite different, making it difficult to combine them for detection directly, as illustrated in Figure 1. After normalization, features from different layers are of the same order of magnitude.

![scales](extras/scales.png)

### C. Anchor Design Strategy
IoU(Intersection over Union) is calculated as Equation (1).

![IoU](extras/IoU.png)

The average IoU and spatial IoU are demonstrated in Figure 2, where the only difference is whether to consider the ***spatial position*** of the anchors.

![AvgSpt](extras/AvgSpt.png)

Moreover, the anchor clustering is carried out for the vehicle detection branch and the license plate detection branch separately. Figure 3 illustrates the anchor distribution mapped back to the original image. The lighter color corresponds to the anchors of the shallower head layers and the darker color corresponds to the anchors of the deeper head layers, where the anchors of shallow layers are small, dense and the anchors of deep layers are large, sparse. For simplicity, only three levels of anchors in the vehicle detection branch are presented, where there should be six in all. As can be seen, the cluster centroids of the vehicle are tall, thin boxes and the cluster centroids of the license plate are short, wide boxes.

![AnchorDistribution](extras/AnchorDistribution.png)
Furthermore, like SSD[2], the anchor priors are placed on multiple feature maps. Let F be the number of feature maps, S<sub>i</sub> be the size of the i-th feature map, A<sub>i</sub> be the number of anchors placed on the i-th feature map, N<sub>anchor</sub> be the total number of anchors. The number of anchors is calculated as Equation (2). For SSD300, the size of six head layers are S={38, 19, 10, 5, 3, 1}, and the anchor number of six head layers are A={4, 6, 6, 6, 4, 4}. From Equation (2), the number of anchors is calculated by 8732=(38x38x4)+(19x19x6)+(10x10x6)+(5x5x6)+(3x3x4)+(1x1x4), et cetera.

![AnchorNumber](extras/AnchorNumber.png)

## REFERENCES
[1]F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proceedings of the 4th International Conference on Learning Representations (ICLR), Y. Bengio and Y. LeCun, Eds., San Juan, Puerto Rico, May 2016.

[2]W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C.-Y. Fu, and A. C. Berg, “SSD: Single shot MultiBox detector,” in Proceedings of the 14th European Conference on Computer Vision (ECCV), Part I, ser. Lecture Notes in Computer Science, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds., vol. 9905. Amsterdam, The Netherlands: Springer, Oct. 2016, pp. 21–37.
