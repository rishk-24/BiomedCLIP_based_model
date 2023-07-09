# BiomedCLIP_based_model

BiomedCLIP is a biomedical vision-language foundation model that is pretrained on PMC-15M, a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, using contrastive learning. It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder, with domain-specific adaptations. It can perform various vision-language processing (VLP) tasks such as cross-modal retrieval, image classification, and visual question answering.


Pre-trained models 
1.	microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 · Hugging Face 

Type – Image classification based on medical image datasets
Description – BiomedCLIP is a biomedical vision-language foundation model that can learn to recognize patterns and concepts specific to the biomedical domain. It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder. It can perform various VLP (Vision Language Processing) tasks such as biomedical image classification, biomedical image retrieval and visual question answering.

Dataset – (not public)
Dataset	Description	Labels
 PMC-15M
curated figure-caption pairs from biomedical research articles in PubMed Central	dataset with 15 million biomedical image-text pairs
