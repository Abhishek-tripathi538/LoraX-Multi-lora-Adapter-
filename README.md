# LoRAX (LoRA Exchange) - BLIP-2 & Open Source Vision-Language Models

A focused system for creating, training, and exchanging LoRA (Low-Rank Adaptation) adapters specifically for **BLIP-2** and other **open-source vision-language models**.

## 🎯 **Core Scripts**

This repository contains **3 specialized Python scripts** that demonstrate different aspects of LoRA adaptation with open-source vision-language models:

### 1. **`blip2_captioning.py`** - BLIP-2 Image Captioning with LoRA
- **Purpose**: Demonstrates image captioning with different styles using LoRA adapters
- **Models**: Salesforce BLIP-2 OPT-2.7B
- **Features**: 
  - Create style-specific adapters (detailed, concise, creative)
  - Compare captioning approaches on same images
  - GPU/CPU compatibility with automatic optimization

### 2. **`vqa_system.py`** - Visual Question Answering Multi-Model System
- **Purpose**: Multi-model VQA system with specialized LoRA adapters
- **Models**: 
  - Salesforce BLIP-2 FLAN-T5-XL (reasoning tasks)
  - Salesforce BLIP-2 OPT-2.7B (general VQA)
  - ViLT (lightweight alternative)
- **Features**:
  - Task-specific adapters (counting, reasoning, classification)
  - Model benchmarking and comparison
  - Multiple open-source VQA approaches

### 3. **`multimodal_exchange.py`** - LoRA Exchange Between Models
- **Purpose**: Demonstrates LoRA adapter creation, exchange, and compatibility
- **Models**:
  - Salesforce BLIP-2 OPT-2.7B
  - Salesforce BLIP-2 FLAN-T5-XL  
  - Salesforce InstructBLIP-Vicuna-7B
- **Features**:
  - Create, save, load, and exchange LoRA adapters
  - Compatibility checking between different models
  - Task specialization (captioning, VQA, instruction-following)

## 🚀 **Quick Start**

### Installation

```bash
# Install required packages
pip install -r requirements_blip2.txt
```

### Run Individual Scripts

```bash
# Image captioning with different styles
python blip2_captioning.py

# Visual question answering system
python vqa_system.py

# Multi-modal LoRA exchange
python multimodal_exchange.py
```

## 📋 **Key Features**

✅ **Pure BLIP-2 & Open Source Models** - No GPT-2 fallbacks  
✅ **Salesforce Models** - BLIP-2 OPT, FLAN-T5, InstructBLIP  
✅ **Additional Open Source** - ViLT for lightweight VQA  
✅ **LoRA Exchange** - Create, save, load, exchange adapters  
✅ **Memory Efficient** - 4-bit quantization for GPU usage  
✅ **CPU/GPU Compatible** - Automatic device detection  

## 🔧 **Models Supported**

### Vision-Language Models
- **BLIP-2 OPT-2.7B**: General purpose vision-language understanding
- **BLIP-2 FLAN-T5-XL**: Advanced reasoning and instruction following  
- **InstructBLIP-Vicuna-7B**: Instruction-tuned vision-language model
- **ViLT**: Lightweight Vision-and-Language Transformer

### Capabilities
- **Image Captioning**: Generate descriptions with different styles
- **Visual Question Answering**: Answer questions about images
- **Instruction Following**: Follow complex visual instructions
- **Multi-Modal Reasoning**: Advanced reasoning about visual content

## ⚙️ **Configuration Examples**

### Image Captioning (blip2_captioning.py)
```python
# Create captioning system with BLIP-2
captioning_system = BLIP2CaptioningSystem("Salesforce/blip2-opt-2.7b")

# Create style-specific adapters
captioning_system.create_captioning_adapter("detailed_captions", "detailed")
captioning_system.create_captioning_adapter("concise_captions", "concise")
captioning_system.create_captioning_adapter("creative_captions", "creative")

# Generate captions with different styles
results = captioning_system.compare_captioning_styles(image_url)
```

### Visual Question Answering (vqa_system.py)
```python
# Initialize multi-model VQA system
vqa_system = VQASystem()

# Create task-specific adapters
vqa_system.create_vqa_adapter("blip2_flan", "counting", "counting")
vqa_system.create_vqa_adapter("blip2_flan", "reasoning", "reasoning")

# Answer questions with specialized adapters
answer = vqa_system.answer_question(image_url, question, "blip2_flan", "counting")
```

### LoRA Exchange (multimodal_exchange.py)
```python
# Initialize exchange system
exchange_system = MultiModalLoRAExchange()

# Create adapters for different tasks
adapter_key = exchange_system.create_adapter(
    "blip2_opt", "image_caption", "captioning", "medium"
)

# Exchange adapters between models
exchange_info = exchange_system.exchange_adapters(adapter1_key, adapter2_key)
```

## 🖥️ **Hardware Requirements**

### Minimum (CPU Only)
- **CPU**: Modern multi-core processor
- **RAM**: 16GB (BLIP-2 models are large)
- **Storage**: 15GB+ free space for models

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080+)
- **RAM**: 32GB+
- **CUDA**: Version 11.8+
- **Storage**: 20GB+ free space

### Model Size Guide
- **BLIP-2 OPT-2.7B**: ~5GB
- **BLIP-2 FLAN-T5-XL**: ~8GB  
- **InstructBLIP-Vicuna-7B**: ~13GB
- **ViLT**: ~400MB (lightweight option)

## 🛠️ **Troubleshooting**

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use 4-bit quantization (automatically enabled on GPU)
   # Reduce batch size to 1
   # Use smaller LoRA rank (r=4 instead of r=8)
   ```

2. **Model Download Errors**
   ```bash
   # Ensure stable internet connection
   # Check Hugging Face model availability
   # Verify sufficient disk space
   ```

3. **CPU Performance Issues**
   ```python
   # BLIP-2 models are optimized for GPU
   # Consider using ViLT for CPU-only setups
   # Use smaller input images (224x224)
   ```

## 📁 **File Structure**

```
LoRAX/
├── blip2_captioning.py      # BLIP-2 image captioning with LoRA
├── vqa_system.py           # Multi-model VQA system  
├── multimodal_exchange.py  # LoRA exchange between models
├── requirements_blip2.txt  # Package dependencies
└── README.md              # This documentation
```

## 🎯 **Use Cases**

### Research Applications
- **Vision-Language Adaptation**: Study how LoRA affects different VL tasks
- **Model Comparison**: Benchmark different BLIP-2 variants
- **Transfer Learning**: Investigate adapter transfer between models

### Practical Applications  
- **Custom Image Captioning**: Train domain-specific captioning styles
- **Specialized VQA**: Create task-specific question answering systems
- **Multi-Modal Chatbots**: Build instruction-following visual assistants

## 📚 **Examples**

Each script includes comprehensive examples and demonstrations. Run them individually to see:

- **Style Transfer**: How LoRA adapters change captioning style
- **Task Specialization**: Adapters optimized for specific VQA tasks  
- **Model Exchange**: Compatibility between different BLIP-2 variants

## 🤝 **Contributing**

This is a focused demonstration of BLIP-2 and open-source vision-language models with LoRA adaptation. Feel free to extend the scripts for your specific use cases.

## 📄 **License**

MIT License - See LICENSE file for details.

## 🔗 **References**

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Salesforce BLIP-2 Models](https://huggingface.co/Salesforce)
- [PEFT Library](https://github.com/huggingface/peft)
