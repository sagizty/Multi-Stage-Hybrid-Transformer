# MSHT: Multi-Stage Hybrid Transformer for ROSE images of pancreatic cancer diagnosing

This is the official page of the MSHT with its full experimental script and record, and everyone can view its whole training process and script. We dedicate to the open-source concept and wish the schoolers can benefit from our release. 

The trained models and the dataset are not available publicly due to the hospital’s requirement.


# background

ROSE is a clinical innovation to diagnose pancreatic cancer. Many time and costs can be saved by EUS-FNA to obtain cell samples and staining faster for the on-site pathologist to draw out the conclusion. However, the requirement of on-site pathologists leads to limitations of expansion of the revolutionary method. Much more life may be saved if the AI system can help the onsite pathologist by simplifying the training and can be used in the hospitals lacking onsite pathologists.

In histology and cytopathology, convolutional neural networks perform robustly and achieve good generalisability by their inductive bias of regional related areas. In the analyzation process of ROSE images, the local feature is also pivotal; shapes and occupation of the cell core area can identify the cancer cells from their counterparts. However, the global features are essential to distinguish the positive samples by their relative size and arrangement. Meanwhile, to perform more robust and constrain well under the limited dataset size is also the barrier we have to face when dealing with the medical dataset. The cutting-edge Transformer module performs excellently in recent CV tasks, and its attention mechanism leads to better global modelling. An idea of hybridising the Transformer and robust CNN backbone to improve the global modelling process can be drawn by the need of the clinical requirement.


# MSHT model

The proposed Multi-stage Hybrid Transformer (MSHT) is designed for pancreatic cancer’s cytopathological images analysis. Along with clinical innovation strategy ROSE, MSHT aims for a faster and pathologist free trend in pancreatic cancer’s diagnoses. The main idea is to concordantly encode features and bias of the early-stage CNN into the global modelling process of the Transformer. MSHT comprises a CNN backbone that generates the feature map from different stages. A focus-guided Decoder structure (FGD) works on global modelling and local attention information hybridising.

![Screen Shot 2021-12-08 at 2 45 42 PM](https://user-images.githubusercontent.com/50575108/145161704-4d589064-9c02-4205-bfad-942eebba11a1.png)


Inspired by the gaze and glance of human eyes, we designed the FGD Focus block to obtain attention guidance. In the Focus block, the feature map from different CNN stages can be transformed to attention guidance of prominent and general information and help the transformer decoder in the global modelling process.



The Focus is stacking up by: 1.An attention block 2.a dual path pooling layer 3. projecting 1x1 CNN 
![Focus](https://user-images.githubusercontent.com/50575108/139060041-0562c141-008a-4af1-aa2c-134dc7a80f59.jpg)

Meanwhile, a new decoder is used to work with the attention guidance from CNN stages. We use the MHGA(multi-head guided attention) to access the prominent and general attention information and encode them inside the transformer modelling process.

![Decoder](https://user-images.githubusercontent.com/50575108/139060071-e34394c1-08a5-40e0-b4a4-9b1032722c64.jpg)

# Experimental result



| Model                   | Acc        | Specificity | Sensitivity | PPV        | NPV        | F1_score   |
| ----------------------- | ---------- | ----------- | ----------- | ---------- | ---------- | ---------- |
| ResNet50                | 95.0177096 | 95.5147059  | 94.1254125  | 92.1702818 | 96.6959145 | 93.1175649 |
| VGG-16                  | 94.9232586 | 95.6617647  | 93.5973597  | 92.4202662 | 96.4380638 | 92.9517884 |
| VGG-19                  | 94.8288076 | 96.0294118  | 92.6732673  | 93.0172654 | 95.9577772 | 92.7757736 |
| Efficientnet_b3         | 93.2939787 | 95.4779412  | 89.3729373  | 91.8015468 | 94.1863486 | 90.5130405 |
| Efficientnet_b4         | 90.9090909 | 94.4117647  | 84.620462   | 89.4313858 | 91.6892225 | 86.9433552 |
| Inception V3            | 93.837072  | 94.4852941  | 92.6732673  | 90.3515408 | 95.8628556 | 91.4941479 |
| Xception                | 94.6871311 | 96.0661765  | 92.2112211  | 92.9104126 | 95.6827139 | 92.5501388 |
| Mobilenet V3            | 93.4356553 | 95.1102941  | 90.4290429  | 91.1976193 | 94.6950621 | 90.7970552 |
| ViT (base)              | 94.498229  | 95.2573529  | 93.1353135  | 91.6291799 | 96.1415203 | 92.3741742 |
| DeiT (base)             | 94.5218418 | 95.0367647  | 93.5973597  | 91.340846  | 96.4118682 | 92.4224823 |
| Swin Transformer (base) | 94.9232586 | 95.1838235  | 94.4554455  | 91.7376454 | 96.8749621 | 93.0308148 |
| MSHT (Ours)             | 95.6788666 | 96.9485294  | 93.3993399  | 94.5449211 | 96.3529107 | 93.9414631 |



# Abalation studies



| Information            | Model                                | Acc        | Specificity | Sensitivity | PPV        | NPV        | F1_score   |
| ---------------------- | ------------------------------------ | ---------- | ----------- | ----------- | ---------- | ---------- | ---------- |
| directly stack         | Hybrid1_384_401_lf25_b8              | 94.8996458 | 95.5882353  | 93.6633663  | 92.2408235 | 96.4483431 | 92.9292015 |
| 3 satge design         | Hybrid3_384_401_lf25_b8              | 94.7343566 | 96.5441176  | 91.4851485  | 93.6616202 | 95.3264167 | 92.5493201 |
| no class token         | Hybrid2_384_No_CLS_Token_401_lf25_b8 | 94.8524203 | 96.25       | 92.3432343  | 93.2412276 | 95.7652112 | 92.7734486 |
| no positional encoding | Hybrid2_384_No_Pos_emb_401_lf25_b8   | 94.7107438 | 96.1029412  | 92.2112211  | 92.9958805 | 95.7084196 | 92.5636149 |
| no attention module    | Hybrid2_384_No_ATT_401_lf25_b8       | 94.5218418 | 95.4411765  | 92.8712871  | 91.9562939 | 96.0234692 | 92.3824865 |
| SE attention module    | Hybrid2_384_SE_ATT_401_lf25_b8       | 94.7107438 | 96.25       | 91.9471947  | 93.2475287 | 95.5635663 | 92.5598981 |
| CBAM attention module  | Hybrid2_384_CBAM_ATT_401_lf25_b8     | 95.1121606 | 95.9558824  | 93.5973597  | 92.8351294 | 96.4240051 | 93.2000018 |
| No PreTrain            | Hybrid2_384_401_lf25_b8              | 95.3010626 | 96.2132353  | 93.6633663  | 93.2804336 | 96.4716772 | 93.4504212 |
| different lr           | Hybrid2_384_401_PT_lf_b8             | 95.3719008 | 96.25       | 93.7953795  | 93.3623954 | 96.5397241 | 93.5582297 |
| MSHT (Ours)            | Hybrid2_384_401_PT_lf25_b8           | 95.6788666 | 96.9485294  | 93.3993399  | 94.5449211 | 96.3529107 | 93.9414631 |



# Imaging results of MSHT

Focus on the interpretability, the MSHT perform well when visualizing its attention area by grad CAM technique.
![Screen Shot 2021-12-08 at 2 48 27 PM](https://user-images.githubusercontent.com/50575108/145161856-ea7758ad-bc03-4b9e-adac-428d3e849725.png)


*  For most cases, as shown in fig(), MSHT can correctly distinguish the samples and focus on the area like the senior pathologists, which outperform most counterparts.

<img width="1297" alt="Screen Shot 2021-11-05 at 3 20 23 PM" src="https://user-images.githubusercontent.com/50575108/140473142-58747866-784f-481d-8057-ff935c9ac380.png">


*  Additionally, the misclassification problem is yet to be overcome, by taking 2 examples.

A few positive samples were misclassified to their negative counterparts.  Compared with senior pathologists, the small number of the cells made MSHT difficult to distinct cancer cells by its arrangement and relative size information. 

<img width="318" alt="Screen Shot 2021-10-30 at 2 34 03 PM" src="https://user-images.githubusercontent.com/50575108/139523238-ead3dd84-8989-4566-9e50-76243d304167.png">

A specific image was misclassified to positive condition by 3 of the 5-fold models. By the analysis of senior pathologists, the reason can be revealed on the fluctuation of the squeezed sample, which misleads MSHT by the shape of the cells.

<img width="322" alt="Screen Shot 2021-10-30 at 2 34 10 PM" src="https://user-images.githubusercontent.com/50575108/139523247-f4b41d45-ac41-4c99-baf7-39439bd35ff2.png">
