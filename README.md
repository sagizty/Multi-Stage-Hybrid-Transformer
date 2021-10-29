
# MSHT: Multi-Stage Hybrid Transformer for pancreatic cancer diagnoisis

This is the official page of the MSHT with its full experimental script and record, everyone can view its whole training process and script. We dedicate to the opensource concept and wish the schoolers can be benefit from our release. 

The trained models and dataset is not aviliable due to the requirement of the hospital.


# background

ROSE is a clinical innovation to diagnosis the pancreatic cancer. Many time and cost can be saved by EUS-FNA to obtain cell sample and staining faster for the on-site pathologist to draw out the conclution. However, the requirement of on-site pathologist leeds to limitations of expand of the revolutionary method. Many more life may be saved if the AI system can be used to help the onsite pathologist by simplizing the training, and can be used in the hospitals lacking of onsite pathologists.

In the field of histology and cytopathology, convelutional neural networks perform robust and achieved good generaliability by its inductive bias of regional related areas. In the analyzation process of ROSE images, the local feature is also pivital, shape and occupation of cell core area can identify the cancer cells from its counterparts. However, the global feature is important to distint the positive samples by its relative size and arrangement. Meanwhile, to perform more robust and constrain well under the limited dataset size is also the barrier we have to face when dealing with the midical dataset. Cutting-edge Transformer module perform excellent in recent CV tasks, its attention machenism lead to better global modeling. An idea of hybriding Transformer along with robust CNN backbone to improve the global modeling process can be draw by the need of the clinical requirement.


# MSHT model

The proposed Multi-stage Hybrid Transformer (MSHT) is designed for pancreatic cancer’s cytopathological images analysis. Along with clinical innovation strategy ROSE, MSHT is aiming for faster and pathologist free trend in pancreatic cancer’s diagnoses. MSHT is made up with a CNN backbone which generating the feature map from different stages and a Focus-guided Decoder structure (FGD structure) works on global modeling and local attention information hybridizing.

![MSHT](https://user-images.githubusercontent.com/50575108/139060018-fb06dab1-25bf-462c-9d29-c37eed1e3e02.jpg)


Inspired by the gaze and glance of humen eyes, we designed the FGD Focus block to obtain the attention guidance. In the Focus block, the feature map from different CNN stage can be transform to attention duidance of prominet and general information and help the transformer decoder in the global modeling process.


The Focus is stack up by: 1.An attention block 2.a dual path pooling layer 3. projecting 1x1 CNN 
![Focus](https://user-images.githubusercontent.com/50575108/139060041-0562c141-008a-4af1-aa2c-134dc7a80f59.jpg)


Meanwhile, a new decodel is used to work with the attention guidance from CNN stages. We use the MHGA(multi-head guided attention) to access the prominent and general attention information and encode them inside the transformer modeling process.
![Decoder](https://user-images.githubusercontent.com/50575108/139060071-e34394c1-08a5-40e0-b4a4-9b1032722c64.jpg)

