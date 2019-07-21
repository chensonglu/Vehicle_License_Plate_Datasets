# Simultaneous End-to-End Vehicle and License Plate Detection with Multi-Branch Attention Neural Network ***Supplementary Materials***

## III. METHODOLOGY
### A. Base Network
The details of the backbone network are shown in Table I without the ReLU activation function, where layers with **bold** font denote the transformed VGG-16 and the remainders represent the extra layers. As for parameters, "k" means kernel size, "s" stride size, and "d" dilation[1] parameters. In addition, "ceil" means rounding up the size if not divisible, so the size of conv3_3 maybe not exactly 2 times of conv4_1. In addition, layers marked with "\*" are candidates for detection head layers.

<div align="center">
<img src="extras/backbone.png" width="40%">
</div>

### B. Detection Branch
The scale of features in different layers may be quite different, making it difficult to combine them for detection directly, as illustrated in Figure 1. After normalization, features from different layers are of the same order of magnitude.

<div align="center">
<img src="extras/scales.png" width="70%">
</div>

### C. Anchor Design Strategy
IoU(Intersection over Union) is calculated as Equation (1).

<div align="center">
<img src="extras/IoU.png" width="50%">
</div>

The average IoU and spatial IoU are demonstrated in Figure 2, where the only difference is whether to consider the ***spatial position*** of the anchors.

<div align="center">
<img src="extras/AvgSpt.png" width="50%">
</div>

Moreover, the anchor clustering is carried out for the vehicle detection branch and the license plate detection branch separately. Figure 3 illustrates the anchor distribution mapped back to the original image. The lighter color corresponds to the anchors of the shallower head layers and the darker color corresponds to the anchors of the deeper head layers, where the anchors of shallow layers are small, dense and the anchors of deep layers are large, sparse. For simplicity, only three levels of anchors in the vehicle detection branch are presented, where there should be six in all. As can be seen, the cluster centroids of the vehicle are tall, thin boxes and the cluster centroids of the license plate are short, wide boxes.

<div align="center">
<img src="extras/AnchorDistribution.png" width="50%">
</div>

Furthermore, like SSD[2], the anchor priors are placed on multiple feature maps. Let F be the number of feature maps, S<sub>i</sub> be the size of the i-th feature map, A<sub>i</sub> be the number of anchors placed on the i-th feature map, N<sub>anchor</sub> be the total number of anchors. The number of anchors is calculated as Equation (2). For SSD300, the size of six head layers are S={38, 19, 10, 5, 3, 1}, and the anchor number of six head layers are A={4, 6, 6, 6, 4, 4}. From Equation (2), the number of anchors is calculated by 8732=(38x38x4)+(19x19x6)+(10x10x6)+(5x5x6)+(3x3x4)+(1x1x4), et cetera.

<div align="center">
<img src="extras/AnchorNumber.png" width="40%">
</div>

### D. Attention and Feature Fusion
Figure 4 demonstrates two feature fusion building blocks from ION[3] and FSSD[4]. The only differece between ION and FPN is the fusion mode, where FPN is element-wise addition and ION is concatenation by channel. FSSD extends a series of pyramid features after the FPN fusion for detection.

<div align="center">
<img src="extras/IONFSSD_new.png" width="100%">
</div>

## IV. EXPERIMENTS
### A. Datasets

<div align="center">
<img src="extras/examples.png" width="100%">
</div>

***VALID*** For simplicity, we name our dataset VALID (Vehicle And LIcense plate Dataset). We employ one auto-mobile data recorder (Ra) to collect videos on the road of a Chinese city\footnote{Zhuhai, China} with the resolution of 720p\footnote{720 $\times$ 1280 (height $\times$ width), 25FPS}. The accumulative total time of videos is about four hours and they are all collected in the daytime. In order to enable data diversity, our data acquisition process was carried out in four days, one hour each day, in different places, such as the city center, suburb, expressway, residential area, etc. Furthermore, we also export ten five-minute videos from another auto-mobile data recorder (Rb), where all videos have the same resolution with videos from Ra. After key-frame extraction every 30 frames, five volunteers worked one day to filter out duplicated or almost invariable images as well as images without vehicles. Finally, 887 images are carefully annotated in one week by five volunteers, where 78 images from Rb are used as the test set and the rest 809 images from Ra are randomly divided into the training set and the validation set by 7:3. Some examples of VALID are illustrated in Figure \ref{fig:VALIDCarOID}. Each license plate must correspond to a vehicle, and a vehicle not always contains a license plate.

***DETROIT*** Open Image Dataset (OID) V4\cite{DBLP:journals/corr/abs-1811-00982} is a dataset of about 9 million images that have been annotated with image-level labels, object bounding boxes, and visual relationships. The dataset spans 600 object classes and the set of all classes are formed as a hierarchy (for instance, "Car" includes "Vehicle registration plate"). Considering that the annotations of the training set of OID are fairly coarse and have unbearable mistakes, we utilize the test set of OID as the training-validation set and the validation set of OID as the test set because of their relatively fine annotations. The training set and the validation set are divided randomly by 7:3. To get the DETROIT (DatasET fRom Open Image daTaset), we first picked out all annotations containing "Vehicle registration plate" and their corresponding images from OID. Secondly, we only preserved the annotations of "Car" and "Vehicle registration plate", as illustrated in Figure \ref{fig:VALIDCarOID}. The images of DETROIT are obtained from the Internet, and the size and aspect ratio vary greatly, where the size (height $\times$ width) ranges from 433 $\times$ 1000 to 4000 $\times$ 6016 and the aspect ratio (width/height) scopes from $\frac12$ to $\frac52$.

***DOC*** The Cars\cite{DBLP:conf/iccvw/Krause0DF13} dataset contains 16185 images of 196 classes of cars, and classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 Coupe. All training images have the class information and bounding box of vehicles. \cite{DBLP:conf/eccv/SilvaJ18} manually annotated the four corner coordinates of the license plate in 105 images selected from the training images of Cars\cite{DBLP:conf/iccvw/Krause0DF13}. We simply get the tightest bounding boxes of the license plates based on the annotations of \cite{DBLP:conf/eccv/SilvaJ18}, and then combine them with the position coordinates of their corresponding vehicles from Cars\cite{DBLP:conf/iccvw/Krause0DF13}. Through this method, we obtain the DOC (Dataset frOm Cars) dataset, which includes 105 images with the bounding boxes of the vehicles and their corresponding license plates, as illustrated in Figure \ref{fig:VALIDCarOID}. 70\% are randomly selected as the training-validation set, and the rest 30\% are used as the test set. The images of DOC are also obtained from the Internet, and the size and aspect ratio varies greatly, where the size (height $\times$ width) ranges from 183 $\times$ 275 to 2592 $\times$ 3888 and the aspect ratio (width/height) scopes from 1 to $\frac83$.

To sum up, the statistics of the three datasets are illustrated in Table \ref{tab:statistics}. There are about 5.2 vehicles and 2.3 license plates per image in VALID, and about 2.05 vehicles and 1.39 license plates per image in DETROIT.

## REFERENCES
[1]F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proceedings of the 4th International Conference on Learning Representations (ICLR), Y. Bengio and Y. LeCun, Eds., San Juan, Puerto Rico, May 2016.

[2]W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C.-Y. Fu, and A. C. Berg, “SSD: Single shot MultiBox detector,” in Proceedings of the 14th European Conference on Computer Vision (ECCV), Part I, ser. Lecture Notes in Computer Science, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds., vol. 9905. Amsterdam, The Netherlands: Springer, Oct. 2016, pp. 21–37.

[3]S. Bell, C. L. Zitnick, K. Bala, and R. Girshick, “Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Las Vegas, NV, USA: IEEE, Jun. 2016, pp. 2874–2883.

[4]Z. Li and F. Zhou, “FSSD: Feature Fusion Single Shot Multibox Detector,” arXiv preprint arXiv:1712.00960, 2017.
