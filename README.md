# Semantic Feature-Wise Transformation Relation Network (SFRN) for Automatic Short Answer Grading
**Reference:**  
This model is discussed in detail in the paper:  
[SFRN: Semantic Feature-Wise Transformation Relation Network for Automatic Short Answer Grading](https://aclanthology.org/2021.emnlp-main.487/)

**Note:**  
The `data` folder contains datasets referenced in the associated paper. However, the model can be used for any classification task. A more complete description of the data can be found on [Penn State's Data Commons](https://www.datacommons.psu.edu/commonswizard/MetadataDisplay.aspx?Dataset=6392).


This repository implements the Semantic Feature-Wise Transformation Relation Network (SFRN), a deep learning-based approach for automatic short answer grading. The system compares student responses with reference answers to assess their correctness, providing a scalable and reliable method for grading in educational settings. The repository also includes supporting models, data handling utilities, and training scripts to facilitate experimentation and deployment.

The repository is organized into the following Python files, each serving a specific purpose in the implementation:

main.py: The main script for training and evaluating the models.

model.py: Defines an LSTMClassifier with pre-trained embeddings, as a simple baseline model.

lstm.py: Defines the LSTMClassifier without pre-trained embeddings, as a simple baseline model.

SFRN_model.py: Contains the implementation of the SFRNModel.

DataModules.py: Handles data loading and preprocessing.

istudio_dict_init.py: Initializes dictionaries with questions and reference answers.

utils.py: Provides utility functions for data processing.

constant.py: Constants / parameters

## main.py

main.py is the primary script for training and testing the models. It performs the following:

Argument Parsing: Configures options such as device selection and checkpoint names.

Model Initialization: Loads the specified model and prepares it for training.

Tokenizer and Data Loading: Initializes a tokenizer (e.g., BERT tokenizer) and processes datasets using the SequenceDataset class from DataModules.py.

Training Process: Implements a loop for forward and backward passes, calculates loss, and optimizes the model. Tracks performance metrics like accuracy, F1 score, and Quadratic Weighted Kappa (QWK).
Validation and Testing: Saves the best-performing model during validation. Loads the best model checkpoint for final evaluation on the test set.

Logging:
Includes detailed logging of predictions for further analysis.

## SFRN_model.py

Implements the SFRNModel, a transformer-based architecture for short answer grading:

Feature Extraction: Uses a pre-trained model like BERT to generate embeddings.

Transformation and Aggregation: Applies feature transformation using g, α, and β functions. Aggregates features across tokens for classification.

Output: An MLP processes the aggregated features to predict class probabilities.

## DataModules.py
The SequenceDataset class processes and prepares the data for training:

Input Construction: Combines student answers, reference answers, and question texts into a single sequence. Tokenizes sequences using a specified tokenizer.

Label Mapping: Maps textual labels to numeric IDs for model compatibility.

Methods:
__len__: Returns dataset size.
__getitem__: Retrieves tokenized inputs and labels for a specific index.

## istudio_dict_init.py
Contains dictionaries for questions and reference answers:

q_context_dict: Provides detailed question contexts.

q_text_dict: Contains concise question prompts.

q_rubric_dict: Lists of correct answers for each question.

part_ref_dict: Partially correct answers.

in_ref_dict: Incorrect answers.

## utils.py

Includes utility functions for preprocessing and batching:

vectorized_data: Converts tokens to indices using a vocabulary mapping.

pad_sequences: Pads sequences to a consistent length for batch processing.

create_dataset: Creates a PyTorch DataLoader for handling batches during training.

sort_batch: Sorts sequences by length, useful for models like LSTM.

## Setup

1. Clone the repository

git clone https://github.com/psunlpgroup/sfrn_analysis.git

cd asrrn

2. Install required packages:
   
pip install -r requirements.txt
