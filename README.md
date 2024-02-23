# DR-TransNet 
------------------------------------------------------------------
Transformer based Douglas-Rachford Unrolling Network for Compressed Sensing
-------------------------------------------------------------------
This repository is for DR-TransNet introduced in the following paper:

Yueming Su, Qiusheng Lian, Dan Zhang and Baoshun Shi, "Transformer based Douglas-Rachford Unrolling Network for Compressed Sensing", Signal Processing: Image Communication, 2024.

Training data can be downloaded at: https://pan.baidu.com/s/1ZAzD95pxVcAz1HztlXPhNw   Extracted code：mupn

Other training models can be downloaded at: https://pan.baidu.com/s/1vkuLolhBNUKQH0rNZKlL5w   Extracted code：99cl

-------------------------------------------------------------------
Abstract

Compressed sensing (CS) with the binary sampling matrix is hardware-friendly and memory-saving in the signal processing field. Existing Convolutional Neural Network (CNN)-based CS methods show potential restrictions in exploiting non-local similarity and lack interpretability. In parallel, the emerging Transformer architecture performs well in modelling long-range correlations. To further improve the CS reconstruction quality from highly under-sampled CS measurements, a Transformer based deep unrolling reconstruction network abbreviated as DR-TransNet is proposed, whose design is inspired by the traditional iterative Douglas-Rachford algorithm. It combines the merits of structure insights of optimization-based methods and the speed of the network-based ones. Therein, a U-type Transformer based proximal sub-network is elaborated to reconstruct images in the wavelet domain and the spatial domain as an auxiliary mode, which aims to explore local informative details and global long-term interaction of the images. Specially, a flexible single model is trained to address the CS reconstruction with different binary CS sampling ratios. Compared with the state-of-the-art CS reconstruction methods with the binary sampling matrix, the proposed method can achieve appealing improvements in terms of Peak Signal to Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) and visual metrics.

--------------------------------------------------------------------
Acknowledgments

This code is implemented via PyTorch with a NVIDIA RTX 2080Ti GPU and Intel i7-9700F CPU @ 3.00GHz； this work is based on some related source codes, including the OPINE-Net contributed by Jian Zhang et al, the Uformer contributed by Zhendong Wang et al, and so on. We gratefully acknowledge the authors for sharing their source codes.
