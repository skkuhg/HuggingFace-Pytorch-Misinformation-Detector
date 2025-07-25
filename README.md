# 🔍 Hugging Face PyTorch Misinformation Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

AI-powered misinformation detection and counter-narrative generation using Hugging Face Transformers and PyTorch. This project features automatic claim summarization, fact-checking, and generation of corrective visual graphics.

## 🌟 Features

- **🔍 Claim Summarization**: Automatically summarize long claims using T5/BART models
- **⚖️ Fact Checking**: Zero-shot and fine-tuned fact verification
- **📝 Counter-Narratives**: Generate factual corrections for false claims
- **🎨 Visual Graphics**: Create "Myth vs Fact" infographics using Stable Diffusion XL
- **🖥️ Offline Mode**: Robust fallback mechanisms for limited connectivity
- **💻 Laptop-Friendly**: CPU-optimized for resource-constrained environments
- **🔄 Complete Pipeline**: End-to-end processing from raw text to final graphics

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/skkuhg/HuggingFace-Pytorch-Misinformation-Detector.git
cd huggingface-pytorch-misinformation-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo:
```python
from graphics_generator import run_demo

# Run demonstration with sample claims
results = run_demo()
```

### Basic Usage

```python
from graphics_generator import MisinformationPipeline

# Initialize the pipeline
pipeline = MisinformationPipeline()

# Process a single claim
result = pipeline.process_claim(
    claim="The Earth is flat and NASA is lying to us",
    style="fact_check",
    save_dir="./outputs"
)

print(f"Verdict: {result['verdict']['verdict']}")
print(f"Correction: {result['correction']}")
# Generated graphic saved to ./outputs/
```

### Batch Processing

```python
# Process multiple claims
claims = [
    "Vaccines cause autism in children",
    "5G towers cause cancer", 
    "Regular exercise is beneficial for health"
]

results = pipeline.process_batch(
    claims=claims,
    style="warning",
    save_dir="./batch_outputs"
)
```

## 🏗️ Architecture

### Core Components

1. **ClaimSummarizer** - Summarizes long claims using T5/BART
2. **ZeroShotFactChecker** - Zero-shot classification with keyword fallback
3. **FineTunedFactChecker** - Fine-tuned DeBERTa for fact verification
4. **VerdictFormatter** - Formats predictions into human-readable verdicts
5. **CounterNarrativeGenerator** - Generates factual corrections using FLAN-T5
6. **GraphicsGenerator** - Creates visual graphics using Stable Diffusion XL
7. **MisinformationPipeline** - End-to-end processing orchestrator

### Visual Styles

The system supports multiple visual styles for generated graphics:

- `fact_check` - Professional fact-checking layout (default)
- `professional` - Corporate/academic style
- `social_media` - Eye-catching social media format
- `educational` - Classroom-appropriate design
- `warning` - High-contrast warning style

## 📊 Model Performance

The pipeline uses several pre-trained models with offline fallbacks:

| Component | Primary Model | Fallback Method | Performance |
|-----------|---------------|-----------------|-------------|
| Summarization | T5-small | Text truncation | Fast, reliable |
| Fact Checking | DeBERTa-v3 | Keyword matching | High accuracy |
| Counter-Narratives | FLAN-T5-small | Template-based | Contextual |
| Graphics | Stable Diffusion XL | Simple graphics | High quality |

## 🔧 Configuration

```python
# Custom configuration
config = {
    'output_dir': './my_outputs',
    'model_cache_dir': './models',
    'device': 'cpu',  # or 'cuda' for GPU
    'batch_size': 4,
    'max_length': 512,
    'num_epochs': 3,
    'learning_rate': 2e-5
}

pipeline = MisinformationPipeline(config=config)
```

## 📁 Project Structure

```
huggingface-pytorch-misinformation-detector/
├── misinformation_detector.py    # Core detection classes
├── graphics_generator.py         # Graphics generation and pipeline
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── examples/                     # Usage examples
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## ⚠️ Limitations & Considerations

### Ethical Considerations
- **Human Oversight**: AI-generated corrections should be reviewed by experts
- **Bias Awareness**: Models may perpetuate training data biases
- **Transparency**: Users should understand AI limitations

### Technical Limitations
- **Dataset Size**: Performance depends on training data quality
- **Model Accuracy**: Not 100% accurate, requires human validation
- **Resource Requirements**: Some features need significant compute resources

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and pre-trained models
- **PyTorch** for the deep learning framework
- **Stability AI** for Stable Diffusion models
- **Research Community** for misinformation detection research

---

**⚠️ Disclaimer**: This tool is for research and educational purposes. AI-generated content should always be validated by domain experts before public use.
