
# 简单介绍idea  版本： 8月18日 19：46
![MSHT](https://user-images.githubusercontent.com/50575108/139060018-fb06dab1-25bf-462c-9d29-c37eed1e3e02.jpg)

The proposed Multi-stage Hybrid Transformer (MSHT) is design for pancreatic cancer’s cytopathological images analysis. Along with clinical innovation strategy ROSE, MSHT is aiming for faster and pathologist free trend in pancreatic cancer’s diagnoses. MSHT is made up with a CNN backbone which generating the feature map from different stages and a Focus-guided Decoder structure (FGD structure) works on global modeling and local attention information hybridizing.


我们的motivation在于希望不同stage卷积的注意力特征带来的对细节的关注能够帮助全局建模的transformer在不同stage上更好的注意到局部特性。具体将通过focus模块引导decoder的cross attention模块，从而在卷积模块考虑了局部特征并增强泛化性的同时，帮助transformer模块在全局建模中获得优势。
![Focus](https://user-images.githubusercontent.com/50575108/139060041-0562c141-008a-4af1-aa2c-134dc7a80f59.jpg)

我们注意到，卷积网络在病理学图像的研究中被广泛应用，由于卷积运算具有的inductive bias他们通常具有较好的泛化性并在下游任务的迁移中表现良好。从计算特性上来说，采用卷积神经网络的模型能够较好的关注局部特征，但在全局特征建模上由于其感受野交互的限制需要依托于层级结构，这样的设计使得其在关注全局特征时局部特征的重要性被削弱了。在rose数据中，全局的细胞分布与细胞间相对位置关系十分重要（引用医学内容..）同时癌症与否的确认也与细胞的细致结构紧密相关（引用医学内容…）。因此全局特征的建模能够更好的适配本任务。

新兴的transformer模型在一系列近期cv文章中表现优异。由于其计算采用全局注意力的特征，transformer模型具有全局建模上更加优异的性能，对于不同位置的目标区域能够更好的综合考虑。为了结合两大类网络的优势，一系列计算机视觉领域的文章进行了融合模型的探究，与该领域不同的是，医学图像领域有限且昂贵的数据难以支撑纯transformer模型实现足够具有泛化性的建模。针对病理学图像分类任务相关研究较少，如何利用有限的数据，针对该数据的特征与领域存在的问题，有效的建立融合模型从而在分类中性能优异并具有泛化性十分具有挑战性。
![Decoder](https://user-images.githubusercontent.com/50575108/139060071-e34394c1-08a5-40e0-b4a4-9b1032722c64.jpg)

为了获得不同stage的特征，以unet为代表的skip connection策略取得了良好效果，在分割任务中被大量采用，然而分割任务的像素级预测对于快速高效且低维度的分类任务来说过于复杂且冗余。我们希望能够在不同stage获得一个较小的feature来辅助后续transformer注意力建模从而以较低成本达到合理的优化。早期的融合模型可以追溯到首篇vision transformer的文章vit中，而后续的一系列文章均为优化特征交互付出了大量努力（介绍分stage交互，logo融合，平行交互融合的模型）
