# BiomedCLIP_based_model

BiomedCLIP is a biomedical vision-language foundation model that is pretrained on PMC-15M, a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, using contrastive learning. It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder, with domain-specific adaptations. It can perform various vision-language processing (VLP) tasks such as cross-modal retrieval, image classification, and visual question answering.
Problem statement with their use cases in medical domain: -
1. VQA (Visual Question Answering)- 
    Problem statement-
The problem statement refers to the task of developing an automated system that can answer questions related to images or visual content. It involves understanding both the visual information in an image and the textual information in a question and generating accurate and meaningful answers.
VQA can be applied in medical domain to assist healthcare professionals in diagnosing and treating patients by combining computer vision and natural language processing techniques to enable machines to answer questions related to visual content, such as images or videos. By applying VQA in the medical field, healthcare providers can leverage this technology to gain insights from medical images, improve diagnosis accuracy, and enhance patient care.

Use cases of how VQA can be applied in medical domain: -
Medical Image Analysis:
          VQA can be used to analyze medical images, including X-rays, CT scans, MRIs and more. By posing questions about specific features or abnormalities in the images, healthcare professionals can receive detailed answers from the VQA system. For instance, a doctor examining a radiology image may have specific questions about the image that can be asked using natural language, such as “What is the size of the tumor?” or “Is there any abnormality in the blood vessels?” instead of manually analyzing the image and measurements, the doctor can input these questions into a VQA system. 
VQA system follows a multi-step process to analyze medical images and provide answers to questions. Working of VQA involves following steps:
1.	Image Preprocessing: The first step is to preprocess the medical image to prepare it for analysis. This may involve resizing the image, normalizing pixel values, and applying image enhancement techniques to improve visibility and highlight relevant features. Preprocessing ensures that the image is in a suitable format for analysis.
2.	Feature Extraction: Once the image is preprocessed, the VQA system extracts relevant visual features from the image. Convolutional Neural Networks(CNNs) are commonly used for this task. CNNs are deep learning models that can automatically learn and extract visual features from images. The CNN process the image and generates a high-dimensional feature representation capturing various visual patterns and structures.

3.	Question Processing: The VQA system then processes the textual question posed by the healthcare professionals or the patient. Natural Language Processing (NLP) techniques are used to analyze the question and understand it semantics. This involves tokenizing the question, removing stop words, and
applying techniques such as stemming or lemmatization to normalize the text.

4.	Fusion of Image and Question: The extracted visual features from the image and the processed question are combined to create a joint representation that captures the relationship between the visual and textual inputs. This joint representation serves as input for the subsequent stages.

5.	Answer Prediction: The joint representation is fed into a classifier or a recurrent Neural Network (RNN) to predict the answer to the question. The classifier is trained on a labeled dataset that pairs questions with their corresponding answers. During training, the model learns the association between visual features, textual context, and the correct answers. The prediction can be a single-word answer, a phrase, or a probability distribution over a predefined set of answer options.

6.	Post-processing and Answer Presentation: After the answer is predicted, post-processing steps can be applied to refine the output. This may involve filtering out irrelevant answers, ranking the answers based on confidence scores, or converting the output into a more human-readable format. The final answer can be presented as test, visual annotations overlaid on the image, or a combination of both.


At training time-   Input – Images, Questions, Answers
At Inference time – Input – Questions, Output – Answers
