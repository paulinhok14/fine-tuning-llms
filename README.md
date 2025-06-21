# 🦙 LLM Fine-Tuning with PEFT & LoRA
### **Efficient Large Language Model Adaptation on Consumer GPUs**
### 🎯 Fine-tune LLMs with HuggingFace, PEFT, LoRA/QLoRA, and BitsAndBytes

---

## 🎬 What is this Project?

This project demonstrates how to efficiently fine-tune Large Language Models (LLMs) using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA and QLoRA, leveraging HuggingFace's Transformers, PEFT, and bitsandbytes for quantization. The workflow is designed to run on consumer hardware (e.g., 4GB VRAM GPUs) by using quantization and offloading strategies, making LLM adaptation accessible to everyone.

You'll learn how to:
- Load and quantize LLMs (e.g., TinyLlama-1.1B) in 8-bit or 4-bit precision
- Apply LoRA/QLoRA for efficient fine-tuning
- Prepare and tokenize custom datasets
- Train and save your own fine-tuned LLM

---

## 🚀 Key Features

- **Quantized Model Loading**: Run LLMs in 8-bit or 4-bit mode to fit on low-VRAM GPUs
- **PEFT with LoRA/QLoRA**: Efficiently fine-tune only a small subset of parameters
- **Custom Dataset Support**: Easily plug in your own prompt/response data
- **Trainer & Accelerate**: Use HuggingFace's SFTTrainer or custom loops for maximum compatibility
- **GPU/CPU Offloading**: Automatically offload layers to CPU if GPU memory is insufficient
- **Reproducible Workflow**: All steps in a single, well-documented Jupyter notebook

---

## 🛠️ Technologies Used

| Category                | Technology                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| 📦 Programming Language | [Python](https://www.python.org/)                                          |
| 🤗 LLMs & Transformers  | [HuggingFace Transformers](https://huggingface.co/transformers/), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl) |
| 🧮 Quantization         | [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)                |
| 🏋️ Training Utilities  | [Accelerate](https://github.com/huggingface/accelerate), [Datasets](https://github.com/huggingface/datasets) |
| 📊 Data Science         | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)           |
| 🧪 Experiment Tracking  | [TQDM](https://tqdm.github.io/)                                            |
| 🖥️ Hardware Support    | CUDA, NVIDIA GPUs (tested on 4GB VRAM)                                     |
| 🔧 Environment         | [python-dotenv](https://pypi.org/project/python-dotenv/)                    |

---

## 📂 Project Structure

```
fine-tuning-llms/
│
├── FineTuning-LLM-PEFT.ipynb      # Main notebook: step-by-step fine-tuning workflow
├── requirements.txt               # All dependencies (see below)
├── training_dataset.jsonl         # Example training data (prompt/response pairs)
├── TinyLlama-1.1b-Chat-FineTuned-v1.0/ # Output directory for your fine-tuned model
├── venv/                          # (Optional) Python virtual environment
└── .gitignore
```

---

## 🏁 How to Run

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd fine-tuning-llms
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up your environment:**
   - Add your HuggingFace token to a `.env` file as `HUGGINGFACE_TOKEN=your_token_here`

4. **Prepare your training data:**
   - Format: JSONL with `prompt` and `response` fields (see `training_dataset.jsonl` for an example).

5. **Run the notebook:**
   - Open `FineTuning-LLM-PEFT.ipynb` in Jupyter or VSCode and follow the steps.

---

## 📝 Workflow Overview

1. **Setup & Environment Checks**: Ensures compatible versions and available GPU/CPU resources.
2. **Data Loading & Formatting**: Loads your dataset and formats it for language modeling.
3. **Tokenizer Preparation**: Loads and configures the tokenizer for your chosen LLM.
4. **Model Loading with Quantization**: Loads the base LLM in 8-bit or 4-bit mode using bitsandbytes.
5. **PEFT/LoRA Configuration**: Applies LoRA adapters for parameter-efficient fine-tuning.
6. **Training**: Fine-tunes the model using HuggingFace's SFTTrainer or a custom loop.
7. **Saving**: Exports your fine-tuned model for later use.

---

## 💡 Tips & Troubleshooting

- **Low VRAM?** Use 4-bit quantization and enable CPU offloading.
- **OOM Errors?** Lower batch size, sequence length, or use `device_map="auto"`.
- **Custom Data:** Make sure your JSONL is properly formatted.
- **Windows Support:** bitsandbytes is supported via special wheels (see notebook for install tips).

---

## 📈 Example: Training Data Format

```json
{"prompt": "What is LoRA?", "response": "LoRA is a parameter-efficient fine-tuning method for LLMs."}
{"prompt": "How do I quantize a model?", "response": "You can use bitsandbytes to load models in 8-bit or 4-bit precision."}
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/)
- [Tim Dettmers (bitsandbytes)](https://github.com/TimDettmers/bitsandbytes)
- [TinyLlama](https://huggingface.co/TinyLlama)

--- 