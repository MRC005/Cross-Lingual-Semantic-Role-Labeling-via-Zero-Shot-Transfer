# Cross-Lingual Semantic Role Labeling via Zero-Shot Transfer

# Project Overview

This project investigates whether Semantic Role Labeling (SRL) trained only on English can generalize to other languages without requiring labeled data in those languages.

The main research question was:

If we fine-tune a multilingual transformer model on English semantic role annotations, can it understand the same semantic structure in Hindi, Tamil, Assamese, or Chinese without any additional training?

This project demonstrates that zero-shot cross-lingual semantic transfer is possible using multilingual pretrained models.

# What is Semantic Role Labeling?

Semantic Role Labeling identifies the roles played by words in a sentence.

Example:

Sentence:
John kicked the ball.

SRL Output:
John → ARG0 (Agent)  
kicked → V (Predicate)  
ball → ARG1 (Object)  

This structure enables systems to answer questions such as:
Who kicked? → John  
What was kicked? → ball  

SRL forms the foundation for semantic search, question answering, and information extraction systems.

# Motivation

Most NLP systems work well for English due to the availability of large annotated datasets.

However, for Indian languages such as:
- Hindi
- Tamil
- Assamese

There is very limited annotated data available.

Creating new labeled datasets manually is:
- Expensive
- Time-consuming
- Not scalable

Instead of creating new labeled corpora, this project explores zero-shot cross-lingual transfer using multilingual transformers.

# Model Selection

Model used:
bert-base-multilingual-cased

Reason for selection:

1. It is pretrained on more than 100 languages.
2. It uses a shared embedding space across languages.
3. It supports Indian languages reasonably well.
4. It is computationally lighter than larger models such as XLM-R large.

Because the model already encodes multiple languages in a shared vector space, fine-tuning it on English allows knowledge to transfer to other languages.

# Dataset Used

Training dataset:
Universal Propositions – English EWT

Details:
- Approximately 40,000 training sentences
- BIO tagging format
- PropBank-style semantic roles

No labeled data from Hindi, Tamil, Assamese, or Chinese was used during training.

This ensures that evaluation on other languages is purely zero-shot.

# Development Process

## Step 1: Initial Local Training Attempt

Initially, training was performed on a personal laptop (MacBook Air).

Problems encountered:
- High CPU usage
- Laptop overheating
- Training extremely slow without GPU
- Risk of system instability

Because token classification over 40,000 sentences with a transformer model is computationally expensive, training on CPU was not practical.

## Step 2: Migration to Google Colab

To overcome hardware limitations, training was moved to Google Colab.

Reasons for using Google Colab:
1. Free access to GPU (Tesla T4).
2. Faster training compared to CPU.
3. Easy environment setup.
4. Ability to experiment without hardware constraints.

However, this introduced new challenges.

# Technical Challenges Faced

## 1. Silent Script Termination in Colab

Problem:
The training script terminated without errors or logs when executed as a standalone Python file.

Cause:
HuggingFace Trainer uses multiprocessing internally. Without a proper main guard, Python may silently exit in certain notebook-based environments.

Solution:
Wrapped training code inside:

if __name__ == "__main__":
    main()

After adding this guard, the script executed correctly.

## 2. Mixed Precision (fp16) Instability

Problem:
Training crashed when fp16=True was enabled on Tesla T4 GPU.

Cause:
Torch and CUDA compatibility issues in the Colab runtime.

Solution:
Disabled mixed precision temporarily:

fp16=False

This stabilized training and allowed completion of 3 epochs.

## 3. Library Version Conflicts

While experimenting, different combinations of:
- torch
- transformers
- accelerate

caused inconsistent behavior.

Stable versions were installed explicitly to ensure compatibility.

## 4. Colab Storage is Temporary

Colab deletes files after session ends.

To prevent losing the trained model:
- The final model was uploaded to Hugging Face Hub.
- Code was version-controlled using GitHub.

This ensured reproducibility and long-term storage.

## 5. Subword Token Fragmentation During Inference

During initial inference tests, names such as “Rahul” appeared partially (e.g., “Ra”).

Cause:
WordPiece tokenization splits words into subword units.

Solution:
Used HuggingFace aggregation:

aggregation_strategy="simple"

This automatically merges subwords into full tokens.

## 6. Long Paragraph Argument Confusion

When testing on long paragraphs with multiple verbs, incorrect role extraction occurred.

Cause:
SRL operates per predicate. Multiple events introduce ambiguity.

Solution:
Implemented a lightweight sentence selection layer:
- Extract keywords from question.
- Select most relevant sentence.
- Apply SRL only to that sentence.

This improved answer accuracy for longer inputs.

# Final System Workflow

1. User inputs a paragraph (any language).
2. User inputs a question (any supported language).
3. System detects question type and maps it to semantic role.
4. Most relevant sentence is selected.
5. SRL model is applied.
6. Tokens matching the semantic role are extracted.
7. Final answer is returned.

# Cross-Lingual Demonstration

Paragraph:
Rahul organized a charity football match.

Questions:

Hindi:
किसने मैच आयोजित किया?

Tamil:
யார் போட்டியை ஏற்பாடு செய்தார்?

Assamese:
কোনে খেলখন আয়োজন কৰিছিল?

Chinese:
谁组织了这场比赛？

All correctly return:
Rahul

This demonstrates zero-shot cross-lingual semantic understanding.

# Results

English F1 Score: approximately 0.25

Although the score is modest, the objective of this project was demonstration of cross-lingual transfer rather than achieving state-of-the-art performance.

The system successfully shows:
- Language-independent semantic structure
- Zero-shot transfer capability
- Multilingual role detection

# Deployment

The trained model is hosted on Hugging Face:

MRC005/cross-lingual-srl

It can be loaded using:

from transformers import pipeline
pipeline("token-classification", model="MRC005/cross-lingual-srl")

This ensures:
- Reproducibility
- Public access
- Permanent storage

# Limitations

- Performance decreases on multi-event paragraphs.
- ARG1 detection weaker than ARG0.
- No predicate disambiguation.
- Limited training epochs.

# Future Improvements

- Increase training epochs.
- Experiment with XLM-RoBERTa.
- Add predicate-aware answer extraction.
- Evaluate formally on Hindi Universal Propositions.

# Conclusion

This project demonstrates that multilingual pretrained transformers enable semantic role transfer across languages without additional labeled data.

By fine-tuning only on English, the model is able to extract meaningful semantic roles in Hindi, Tamil, Assamese, and Chinese.

This approach provides a scalable and cost-effective solution for extending NLP systems to low-resource languages.
