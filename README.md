# Open-LLMs Documentation

## Introduction

Open-LLMs is a collection of language models and tools designed to facilitate various natural language processing tasks. In this document, we'll discuss how to utilize Open-LLMs for text generation tasks using Llama2, a powerful conversational AI model.

## Requirements

To use Open-LLMs effectively, ensure you have the following:

- Access to the internet for model retrieval and interaction.
- Python environment with the required libraries installed:
  - `transformers`
  - `einops`
  - `accelerate`
  - `langchain`
  - `bitsandbytes`

## Getting Started

### Obtaining Llama2

1. Visit the [Official Llama2 website](https://llama.meta.com/llama2) for more information on the Llama2 model.
2. You can access the Llama2 model directly from its GitHub repository: [meta-llama/llama](https://github.com/meta-llama/llama).

### Setting Up Environment

1. Ensure you have access to a compatible GPU. You can verify GPU availability by running `!nvidia-smi`.
2. Install required Python libraries using `pip install -q transformers einops accelerate langchain bitsandbytes`.

### Authentication

1. You'll need to authenticate using Hugging Face CLI. Follow the instructions provided by the CLI to log in.
2. Install `git config --global credential.helper store` for token storage if prompted.

## Usage

### Importing Libraries

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers
import torch

## Loading Llama2 Model

# Define the model name
model = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# Load model pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

## Initializing Llama2 Pipeline

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

## Generating Text

# Define prompt
prompt = "What would be a good name for a company that makes colorful socks"

# Generate response
print(llm(prompt))

# Define prompt for Indian restaurant name suggestion
prompt = "I want to open a restaurant for Indian food. Suggest me a name for this."

# Generate restaurant name suggestion
print(llm(prompt))

**## Conclusion**

Open-LLMs, particularly leveraging Llama2, provides a powerful platform for text generation tasks with easy access to state-of-the-art language models, you can accomplish various natural language processing tasks effectively.
Explore further possibilities and enjoy using Open-LLMs for your projects!

