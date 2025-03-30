# BioMedGPT: Medical Question Answering with Structured Reasoning and Verification

![Medical AI](https://img.shields.io/badge/Field-Medical%20AI-blue)
![License](https://img.shields.io/badge/License-MIT-green)

BioMedGPT is a robust system for medical question answering that combines structured reasoning with automated verification. It integrates multiple medical QA datasets and fine-tunes language models to produce verifiable answers in a standardized XML format.

## Features

- **Unified Medical Dataset**: Integrates 4 major medical QA benchmarks:
  - MedQA-USMLE (USMLE-style questions)
  - MedMCQA (Indian medical entrance questions)
  - PubMedQA (Evidence-based biomedical QA)
  - MMLU (General-domain verification questions)
  
- **Structured Reasoning**: Enforces XML-based response format:
  ```xml
  <think>...step-by-step reasoning...</think>
  <answer>...final verified answer...</answer>
  ```

- **Medical Answer Verifier**: Automated validation system that checks:
  - Response format compliance
  - Answer correctness against ground truth
  - Clinical plausibility of reasoning

- **Efficient Training**: Utilizes modern techniques:
  - 4-bit quantization via `bitsandbytes`
  - LoRA adapters for parameter-efficient tuning
  - vLLM for high-throughput inference

## Installation

```bash
# Core dependencies
pip install datasets unsloth vllm

# Optional for verification
pip install wandb regex

# For medical dataset processing
pip install pandas numpy pyarrow
```

## Usage

### 1. Dataset Preparation

```python
from datasets import load_dataset

# Load individual datasets
medqa = load_dataset("Neelectric/MedQA-USMLE")
medmcqa = load_dataset("openlifescienceai/medmcqa")
pubmed_qa = load_dataset("bigbio/pubmed_qa")
mmlu = load_dataset("TIGER-Lab/MMLU-Pro")

# Create unified dataset
verifiable_dataset = create_verifiable_dataset()
```

### 2. Model Training

```python
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    max_seq_length=1024,
)

# GRPO Training Configuration
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_correct_answer, reward_strict_format],
    train_dataset=formatted_dataset,
    # ... additional training params
)
trainer.train()
```

### 3. Inference & Verification

```python
from medical_verifier import MedicalVerifier

verifier = MedicalVerifier()
sample = random.choice(dataset)

# Generate response
response = model.generate(sample["prompt"])

# Verify answer
is_valid = verifier.verify(
    response, 
    ground_truth=sample["y_star"]
)
```

## Example Output

**Input:**
```
A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...
```

**Model Response:**
```xml
<think>
Patient presents with UTI symptoms during pregnancy. Key considerations:
- Pregnancy-safe antibiotics required
- Nitrofurantoin is Category B in 2nd trimester
- Avoid fluoroquinolones due to fetal risk
- Asymptomatic bacteriuria screening important
</think>
<answer>
Nitrofurantoin
</answer>
```

**Verification Result:** ✅ Format valid | Answer correct

## Acknowledgments

- Medical datasets from Hugging Face
- Training infrastructure powered by Unsloth
- vLLM for optimized inference
- Llama 3.1 model by Meta AI
