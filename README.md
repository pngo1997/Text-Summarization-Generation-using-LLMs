# 📝 Text Summarization using Encoder-Decoder Models & Large Language Models  

## 📜 Overview  
This project explores **text summarization** using multiple approaches:  

1. **Pre-trained Encoder-Decoder Models** (T5) for text summarization.  
2. **Fine-tuning a T5 model** on a new dataset for improved summarization.  
3. **Prompt-based Summarization using Large Language Models (LLMs)** for flexible and adaptive text generation.  

📌 **Key Concepts Covered**:  
- **Encoder-Decoder Models for Summarization**  
- **Model Evaluation Metrics**: ROUGE, Perplexity, BERTScore  
- **Fine-Tuning Transformer Models with Hugging Face**  
- **Prompt Engineering for LLMs**
  
## 🚀 1️⃣ Implementation Details  

### **Summarization using Pre-trained Models**  
✅ **Models Used**:  
- `"google-t5/t5-small"` (general T5 model)  
- `"ubikpt/t5-small-finetuned-cnn"` (fine-tuned on CNN-DailyMail)  

✅ **Dataset**:  
This project uses the **CNN-DailyMail Dataset**, available on Hugging Face:  
🔗 [CNN-DailyMail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)  

✅ **Processing Steps**:  
- Use the **article** column as input text and the **highlights** column as the reference summary.  
- Tokenize using T5’s tokenizer and ensure **PyTorch tensor format**.  

✅ **Evaluation Metrics**:  
- **ROUGE-1 & ROUGE-2** (Lexical overlap)  
- **Perplexity** (Fluency measure, using GPT-2)  
- **BERTScore** (Semantic similarity)  

📌 **Comparison**: Generated summaries from both models are compared against ground truth summaries.  

### **Fine-Tuning a T5 Model for Summarization**  

✅ **Model Training Steps**:  
- Use `"google-t5/t5-small"`  
- **Preprocess data** (clean text, tokenize, create PyTorch DatasetDict).  
- **Train for 3 epochs** using Hugging Face's `Seq2SeqTrainer`.  
- **Evaluate on test set** and compare results.  

✅ **Metrics Computed**:  
- **ROUGE scores**  
- **Perplexity**  
- **BERTScore**  

📌 **Comparison**: Performance of the **fine-tuned T5 model** is compared with the pre-trained T5 model.  

### **Prompt-Based Summarization using Large Language Models**  
✅ **LLMs Used**:  
🔗 [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

✅ **Prompt Engineering**:  
- **Craft multiple prompts** for generating summaries.  
- **Experiment with few-shot learning** (providing examples in prompts).  
- **Evaluate how prompt changes affect summary quality**.  

📌 **Comparison**:  
- Analyze summaries from different LLMs & prompts.  
- Compare with fine-tuned T5 model outputs.  

## 📊 2️⃣ Results & Observations  

### **Performance Metrics**  
![Metrics](https://github.com/pngo1997/Images/blob/main/HF.png)  

📌 **Key Insights**:  
- Overall, fine-tuned model performs better on all metrics, which is what expected since fine-tuning would help the model generate more accurate and relevant summaries. The higher precision and F1 scores suggest an improvement in the relevance and completeness of generated content. The drop in perplexity also indicates that the fine-tuned model is more confident in its predictions. 
Additionally. Optuna was applied to try with different parameters and came to conclusion that the simpler the better. Set max_new_tokens and max_targetLength to 64 because the maximum word count of 'description' per observation is 28 tokens –do not want the model to over generate. 
- **Fine-tuning improves summary quality** (higher ROUGE & lower Perplexity).  
- **LLMs generate better summaries than pre-trained models** when prompted well.  
- **Prompt design significantly affects summary coherence & informativeness**.  

## 📌 Summary  
✅ Compared pre-trained and fine-tuned T5 summarization models.

✅ Fine-tuned T5 on a new dataset for improved results.

✅ Explored prompt-based summarization using SOTA LLMs.

✅ Analyzed impact of different evaluation metrics.
