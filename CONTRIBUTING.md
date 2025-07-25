# Contributing to Hugging Face PyTorch Misinformation Detector

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve misinformation detection and counter-narrative generation.

## 🤝 How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/huggingface-pytorch-misinformation-detector.git
cd huggingface-pytorch-misinformation-detector
```

### 2. Set Up Development Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

3. Install development dependencies:
```bash
pip install pytest black flake8 mypy
```

### 3. Make Your Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and add tests
3. Run tests to ensure everything works:
```bash
pytest tests/
```

4. Run code formatting:
```bash
black *.py examples/*.py
flake8 *.py examples/*.py
```

### 4. Submit a Pull Request

1. Commit your changes:
```bash
git add .
git commit -m "Add: description of your changes"
```

2. Push to your fork:
```bash
git push origin feature/your-feature-name
```

3. Create a Pull Request on GitHub

## 📝 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant error messages or logs

### 💡 Feature Requests

For new features, please include:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples or references

### 🔧 Code Contributions

We welcome contributions in these areas:

#### Core Functionality
- Improved fact-checking algorithms
- New model integrations
- Performance optimizations
- Better error handling

#### Graphics Generation
- New visual styles
- Enhanced graphics quality
- Support for different image formats
- Accessibility improvements

#### Documentation
- API documentation improvements
- Tutorial enhancements
- Code examples
- Translation to other languages

#### Testing
- Unit tests for existing code
- Integration tests
- Performance benchmarks
- Edge case testing

## 🧪 Development Guidelines

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** for better code documentation
- **Docstrings** for all public functions and classes

Example:
```python
def process_claim(self, claim: str, style: str = 'fact_check') -> Dict[str, Any]:
    """
    Process a single claim through the misinformation detection pipeline.
    
    Args:
        claim: The text claim to analyze
        style: Visual style for generated graphics
        
    Returns:
        Dictionary containing analysis results and generated content
        
    Raises:
        ValueError: If claim is empty or invalid
    """
    pass
```

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Include both unit tests and integration tests
- Test edge cases and error conditions

Example test structure:
```python
def test_claim_processing():
    """Test basic claim processing functionality."""
    pipeline = MisinformationPipeline()
    result = pipeline.process_claim("Test claim")
    
    assert 'verdict' in result
    assert 'confidence' in result['verdict']
    assert result['verdict']['confidence'] >= 0.0
```

### Documentation

- Add docstrings to all public functions
- Update README.md for significant changes
- Include examples for new features
- Comment complex algorithms

## 🚨 Important Considerations

### Ethical Guidelines

This project deals with sensitive topics. Please ensure:

- **No Bias Introduction**: Avoid introducing political, cultural, or personal biases
- **Factual Accuracy**: Base corrections on reliable, scientific sources
- **Transparency**: Clearly indicate AI-generated content limitations
- **Responsible Use**: Consider potential misuse of the technology

### Privacy and Safety

- Never include personal information in test data
- Avoid processing sensitive or private content
- Respect data protection regulations
- Consider security implications of new features

### Model and Data Considerations

- Use only publicly available, ethically sourced models
- Respect model licensing terms
- Avoid training on biased or harmful datasets
- Document model limitations and potential biases

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] No personal information in code/tests
- [ ] PR description clearly explains changes
- [ ] Related issues are referenced

## 🏷️ Issue Labels

We use these labels to organize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or improvement  
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - General questions about usage
- `wontfix` - Issue that won't be addressed

## 🎯 Priority Areas

We especially welcome contributions in:

1. **Model Performance**: Better accuracy, faster processing
2. **Offline Capabilities**: Improved fallback mechanisms
3. **Multilingual Support**: Non-English language support
4. **Accessibility**: Making graphics more accessible
5. **Mobile Optimization**: Lighter models for mobile devices

## 📚 Resources

### Learning Materials
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Misinformation Research Papers](https://scholar.google.com/scholar?q=misinformation+detection)

### Development Tools
- [VS Code](https://code.visualstudio.com/) with Python extension
- [GitHub Desktop](https://desktop.github.com/) for Git GUI
- [Jupyter Notebooks](https://jupyter.org/) for experimentation

## 🙋‍♀️ Getting Help

If you need help:

1. Check existing [Issues](https://github.com/skkuhg/huggingface-pytorch-misinformation-detector/issues)
2. Search [Discussions](https://github.com/skkuhg/huggingface-pytorch-misinformation-detector/discussions)
3. Create a new issue with the `question` label
4. Contact maintainers: [ahczhg@gmail.com](mailto:ahczhg@gmail.com)

## 🏆 Recognition

Contributors will be:
- Added to the project's contributors list
- Mentioned in release notes for significant contributions
- Invited to become maintainers for exceptional contributions

## 📄 Code of Conduct

By participating in this project, you agree to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different opinions and approaches
- Follow ethical guidelines for AI development

Violations may result in removal from the project.

---

Thank you for contributing to making the internet a more trustworthy place! 🌐✨