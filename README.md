# 🦙 LLM Fine-Tuning with PEFT & LoRA
### **Efficient Large Language Model Adaptation on Consumer GPUs (NVIDIA GTX 1050 4GB VRAM)**
### 🎯 Fine-tune LLMs with HuggingFace, PEFT, LoRA/QLoRA, and BitsAndBytes

<img src="imgs/finetuning.png" alt="Finetuning illustration" width="650"/>

<small>*Source: [Neo4j Blog - Fine-tuning vs RAG](https://neo4j.com/blog/developer/fine-tuning-vs-rag/)*</small>

## 🎬 What is this Project?

This project demonstrates how I performed a method of stateful retraining (fine-tuned) Large Language Models (LLMs) using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA and QLoRA, leveraging HuggingFace's Transformers, PEFT, and bitsandbytes for quantization. The workflow is designed to run on consumer hardware (e.g., my 4GB VRAM GPU) by using quantization on a small model (TinyLlama1.1b-Chat), and offloading strategies, making LLM adaptation accessible to everyone.

You'll learn how to:
- Load and quantize LLMs (e.g., TinyLlama-1.1B) in 8-bit or 4-bit precision
- Apply LoRA/QLoRA for efficient fine-tuning
- Prepare and tokenize custom datasets
- Train and save your own fine-tuned LLM
- Generate domain-specific training data from glossaries
- Deploy fine-tuned models with Docker


## 🔄 RAG vs Fine-Tuning: Understanding the Difference

When working with Large Language Models, you have two main approaches to adapt them to your specific domain or use case: **Retrieval-Augmented Generation (RAG)** and **Fine-tuning**. Each approach has its strengths and is suited for different scenarios.

### **Fine-tuning (This Project's Focus)**
Fine-tuning involves **modifying the model's internal parameters** to learn new patterns, behaviors, or knowledge. In this project, we use Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA to efficiently adapt the model:

- **🔄 Model Changes**: Directly modifies the model's weights and parameters
- **📚 Knowledge Integration**: Embeds new knowledge directly into the model's parameters
- **⚡ Inference Speed**: Fast inference as all knowledge is internalized
- **🎯 Domain Adaptation**: Excellent for learning specific styles, formats, or domain-specific knowledge
- **💾 Memory Efficient**: Uses techniques like LoRA to minimize parameter changes
- **🛠️ Use Cases**: Custom assistants, domain-specific models, style adaptation

### **RAG (Retrieval-Augmented Generation)**
RAG keeps the base model unchanged and **retrieves relevant information from external sources** during inference:

<img src="imgs/rag.png" alt="RAG" width="650"/>

<small>*Source: [Neo4j Blog - Fine-tuning vs RAG](https://neo4j.com/blog/developer/fine-tuning-vs-rag/)*</small>

- **📖 External Knowledge**: Uses vector databases to retrieve relevant documents
- **🔄 No Model Changes**: Base model remains unchanged
- **📊 Up-to-date Information**: Can access current or frequently updated information
- **🔍 Contextual Responses**: Provides source citations and context
- **⚙️ Complex Setup**: Requires embedding models, vector databases, and retrieval systems
- **🛠️ Use Cases**: Document Q&A, research assistants, knowledge bases

> **💡 Real-World Example**: For a practical implementation of RAG architecture, check out [H.O.L.M.E.S. - AI Agent for Global Spare Parts](https://github.com/paulinhok14/rag-chatbot), a complete RAG web application developed as a virtual assistant for business collaborators using LangChain, Ollama, and FAISS vector store.

### **When to Use Each Approach?**

| Aspect | Fine-tuning | RAG |
|--------|-------------|-----|
| **Knowledge Type** | Static, domain-specific | Dynamic, frequently updated |
| **Setup Complexity** | High | Moderate |
| **Inference Speed** | Fast | Slower (due to retrieval) |
| **Memory Usage** | Higher (model changes) | Lower (external storage) |
| **Update Frequency** | Requires retraining | Easy to update documents |
| **Cost** | One-time training cost | Ongoing retrieval costs |

This project focuses on **fine-tuning** as it's particularly effective for creating specialized models that can understand and respond to specific domains, formats, or styles while maintaining the efficiency and accessibility needed for consumer hardware.

## 🚀 Key Features

- **Quantized Model Loading**: Run LLMs in 8-bit or 4-bit mode to fit on low-VRAM GPUs
- **PEFT with LoRA/QLoRA**: Efficiently fine-tune only a small subset of parameters
- **Custom Dataset Support**: Easily plug in your own prompt/response data
- **Domain-Specific Training**: Generate training data from glossaries and technical documentation
- **Trainer & Accelerate**: Use HuggingFace's SFTTrainer for maximum compatibility
- **GPU/CPU Offloading**: Automatically offload layers to CPU if GPU memory is insufficient
- **Reproducible Workflow**: All steps in a single, well-documented Jupyter notebook
- **Docker Support**: Containerized deployment for consistent environments
- **Model Merging**: Automatically merge LoRA adapters with base model for deployment


## 🛠️ Technologies Used

| Category                | Technology                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| 📦 Programming Language | [Python 3.12](https://www.python.org/)                                     |
| 🤗 LLMs & Transformers  | [HuggingFace Transformers 4.53.0](https://huggingface.co/transformers/), [PEFT 0.15.2](https://github.com/huggingface/peft), [TRL 0.18.2](https://github.com/huggingface/trl) |
| 🦙 Base Model           | [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| 🧮 Quantization         | [bitsandbytes 0.46.0](https://github.com/TimDettmers/bitsandbytes)         |
| 🏋️ Training Utilities  | [Accelerate 1.8.1](https://github.com/huggingface/accelerate), [Datasets 3.6.0](https://github.com/huggingface/datasets) |
| 🔥 Deep Learning        | [PyTorch 2.2.1+cu121](https://pytorch.org/), [CUDA 12.1](https://developer.nvidia.com/cuda-toolkit) |
| 📊 Data Science         | [Pandas 2.3.0](https://pandas.pydata.org/), [NumPy 1.26.4](https://numpy.org/) |
| 🧪 Experiment Tracking  | [TQDM 4.67.1](https://tqdm.github.io/), [MLflow 3.1.0](https://mlflow.org/) |
| 🐳 Containerization     | [Docker](https://www.docker.com/)                                          |
| 🔧 Environment         | [python-dotenv 1.1.0](https://pypi.org/project/python-dotenv/)             |
| 🖥️ Hardware Support    | NVIDIA GPUs (tested on GTX 1050 4GB VRAM)                                 |


## 📂 Project Structure

```
fine-tuning-llms/
│
├── 📓 notebooks/
│   └── FineTuning-LLM-PEFT.ipynb      # Main notebook: step-by-step fine-tuning workflow
│
├── 📁 src/
│   ├── glossary.py                     # Domain-specific glossary for training data generation
│   └── questions_generator.py          # Script to generate Q&A pairs from glossary
│
├── 📁 data/
│   └── training_dataset.jsonl          # Generated training data (prompt/response pairs)
│
├── 📁 models/
│   ├── TinyLlama-1.1b-Chat-FineTuned-v1.0/     # Training checkpoints
│   └── TinyLlama-1.1b-Chat-FineTuned-v1.0-merged/ # Final merged model for deployment
│
├── 📁 imgs/
│   ├── finetuning.png                  # Fine-tuning visualization
│   └── rag.png                         # RAG architecture visualization
│
├── 📄 requirements.txt                 # All dependencies with specific versions
├── 🐳 Dockerfile                       # Container configuration
└── 📄 .gitignore                       # Git ignore rules
```


## 🏁 How to Run

### **Option 1: Local Development**

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd fine-tuning-llms
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment:**
   - Add your HuggingFace token to a `.env` file as `HUGGINGFACE_TOKEN=your_token_here`

4. **Generate training data (optional):**
   ```bash
   python src/questions_generator.py
   ```

5. **Run the notebook:**
   - Open `notebooks/FineTuning-LLM-PEFT.ipynb` in Jupyter or VSCode and follow the steps.

### **Option 2: Docker Deployment**

1. **Build the Docker image:**
   ```bash
   docker build -t fine-tuning-llms .
   ```

2. **Run the container:**
   ```bash
   docker run -it --gpus all -v $(pwd):/app fine-tuning-llms
   ```

## 📝 Workflow Overview

1. **Setup & Environment Checks**: Ensures compatible versions and available GPU/CPU resources
2. **Data Generation**: Creates domain-specific training data from glossaries using `questions_generator.py`
3. **Data Loading & Formatting**: Loads the dataset and formats it for language modeling
4. **Tokenizer Preparation**: Loads and configures the tokenizer for TinyLlama-1.1B-Chat
5. **Model Loading with Quantization**: Loads the base LLM in 8-bit mode using bitsandbytes
6. **PEFT/LoRA Configuration**: Applies LoRA adapters for parameter-efficient fine-tuning
7. **Training**: Fine-tunes the model using HuggingFace's SFTTrainer
8. **Model Merging**: Merges LoRA adapters with the base model for deployment
9. **Saving**: Exports the fine-tuned model for later use


## 🎯 Domain-Specific Training

This project includes a specialized training data generation system for **aviation and aerospace domains** using mock data:

- **Aviation Segments**: Executive, Commercial, and Military aviation terminology
- **Supply Chain**: Logistics, procurement, and inventory management concepts
- **SAP Transactions**: Common SAP system transactions and their purposes
- **Quality Management**: Lean, Six Sigma, and quality control methodologies
- **Engineering**: Aircraft parts, maintenance, and technical specifications

The training data is automatically generated from a comprehensive glossary, ensuring consistent and accurate domain knowledge.


## 💡 Tips & Troubleshooting

### **Hardware Optimization**
- **Low VRAM (4GB)** Using 8-bit quantization and enable CPU offloading for my limited resource
- **OOM Errors?** Lower batch size, sequence length, or use `device_map="auto"`
- **Windows Support:** bitsandbytes is supported via special wheels (see notebook for install tips)

### **Training Optimization**
- **Custom Data:** Make sure your JSONL is properly formatted with `prompt` and `response` fields
- **Model Selection:** TinyLlama-1.1B is optimized for consumer GPUs; larger models require more VRAM
- **LoRA Configuration:** Adjust `r` (rank) and `lora_alpha` based on your specific needs

### **Common Issues**
- **Version Conflicts:** Ensure transformers, peft, and trl versions are compatible
- **CUDA Issues:** Verify NVIDIA drivers and CUDA installation
- **Memory Management:** Use gradient checkpointing and proper garbage collection

## 📈 Example: Training Data Format

The project generates training data in JSONL format:

```json
{"prompt": "What does 'ATP' mean in the Embraer context?", "response": "ATP stands for 'Available to Promise'. It represents physical stock minus overdue orders, essential for delivery planning."}
{"prompt": "What's the difference between PO and POI in Embraer's SAP system?", "response": "PO is an external Purchase Order from a customer, while POI is an internal order for transfers between Embraer sites."}
```

### **Generated Dataset Statistics**
- **Total Entries**: 105 Q&A pairs
- **Domain Coverage**: Aviation, supply chain, logistics, parts
- **Format**: JSONL with prompt/response structure
- **Source**: Automated generation from domain glossary


## 🚀 Model Performance

The fine-tuned model demonstrates significant improvements in domain-specific knowledge:

- **Training Loss**: Reduced from 3.5 to 0.28 over 200 steps
- **Domain Accuracy**: Improved responses for aviation and aerospace terminology
- **Parameter Efficiency**: Only 0.1% of parameters are trainable (1.1M out of 1.1B)
- **Memory Usage**: Optimized for 4GB VRAM GPUs
- **Results**: The output of the merged model in the same 5 test questions are not so good yet. I'll maybe generate few more (going to 500 instead of 105) different and random task questions to fine-tune base TinyLlama model.

---