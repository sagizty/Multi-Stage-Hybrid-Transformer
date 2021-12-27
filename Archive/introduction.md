The full experiment results & CAM imaging samples


In the 5-fold training process, the MSHT achieves an average of 97.53%, 98.07%, 96.56%, 96.54%, and 98.08% for Acc, Spe, Sen, PPV, and NPV. Meanwhile, the average performance of these indicators during the validating process are 94.37%, 96.69%, 90.20%, 93.93%, and 94.67%. In the independent test dataset, 95.68 % of images are correctly classified, and MSHT achieved 96.95%, 93.40%, 94.54%, and 96.35% for Spe, Sen, PPV, and NPV.

Due to the limited work of this field, we evaluated the proposed MSHT with seven widely applied state-of-the-art CNNs including: ResNet50 [28] (2016), VGG-16, VGG-19 [26] (2014), EfficientNet_b3 [32] (2019), Inception-V3 [29] (2016), Xception [31] (2017) and MobileNet-V3 [33] (2019).

As the first work to introduce the Transformer into ROSE image analysis, three cutting-edge Transformer-based models (vision transformers) from the computer vision field were used as the comparison models responsibly, including ViT [24] (2020), DeiT [44] (2020), and Swin Transformer [45] (2021). It should be noted that Transfer learning was used for all models with the official weight of models pre-trained on the ImageNet [46]. The models were compared with the same criteria on the test dataset after they all converged at the same hyperparameter setting. 
