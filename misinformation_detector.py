"""
Misinformation Detection and Counter-Narrative Generation Pipeline

This module provides a complete pipeline for detecting misinformation and generating
counter-narratives with visual graphics using Hugging Face Transformers and PyTorch.

Key Components:
- ClaimSummarizer: Summarizes long claims using T5/BART models
- ZeroShotFactChecker: Performs zero-shot fact checking
- FineTunedFactChecker: Fine-tuned model for fact verification
- VerdictFormatter: Formats prediction results
- CounterNarrativeGenerator: Generates corrective narratives
- GraphicsGenerator: Creates myth vs fact visual graphics
- MisinformationPipeline: End-to-end processing pipeline
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Try importing transformers with fallback
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
        DataCollatorWithPadding, pipeline
    )
    transformers_available = True
except ImportError:
    transformers_available = False
    print("⚠️ Transformers not available. Install with: pip install transformers")

# Try importing datasets with fallback
try:
    from datasets import Dataset
    datasets_available = True
except ImportError:
    datasets_available = False
    print("⚠️ Datasets not available. Install with: pip install datasets")


class ClaimSummarizer:
    """Summarizes long claims using T5 or BART models with offline fallback."""
    
    def __init__(self, model_candidates=None):
        self.model_candidates = model_candidates or [
            "t5-small",
            "facebook/bart-base", 
            "google/flan-t5-small"
        ]
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the best available summarization model."""
        print("🔧 Loading claim summarizer...")
        
        for model_name in self.model_candidates:
            try:
                print(f"🔧 Loading summarization model: {model_name}")
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # CPU only for laptop compatibility
                )
                print(f"✅ Successfully loaded: {model_name}")
                return
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        print("   ⚠️ No summarization models could be loaded (offline mode or connectivity issues)")
        print("   📝 Using simple text truncation fallback")
    
    def summarize_claims(self, texts):
        """Summarize a list of claim texts."""
        summaries = []
        
        for text in texts:
            if self.pipeline:
                try:
                    # Prepare input with T5 prefix
                    input_text = f"summarize: {text}"
                    result = self.pipeline(
                        input_text,
                        max_length=100,
                        min_length=20,
                        do_sample=False,
                        truncation=True
                    )
                    summary = result[0]['generated_text']
                except Exception as e:
                    print(f"⚠️ Summarization failed: {e}")
                    # Fallback to truncation
                    summary = text[:150] + "..." if len(text) > 150 else text
            else:
                # Simple truncation fallback
                summary = text[:150] + "..." if len(text) > 150 else text
            
            summaries.append(summary)
        
        return summaries


class ZeroShotFactChecker:
    """Zero-shot fact checking with keyword-based fallback."""
    
    def __init__(self, model_candidates=None):
        self.model_candidates = model_candidates or [
            "microsoft/DialoGPT-medium",
            "distilbert-base-uncased",
            "facebook/bart-large-mnli"
        ]
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the best available classification model."""
        print("🔧 Loading zero-shot fact checker...")
        
        for model_name in self.model_candidates:
            try:
                print(f"🔧 Loading zero-shot model: {model_name}")
                if transformers_available:
                    self.pipeline = pipeline(
                        "zero-shot-classification",
                        model=model_name,
                        device=-1  # CPU only
                    )
                    print(f"✅ Successfully loaded: {model_name}")
                    return
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        print("   ⚠️ No zero-shot models could be loaded (offline mode or connectivity issues)")
        print("   📝 Using keyword-based classification fallback")
    
    def check_claims(self, claims, labels=None):
        """Check if claims are true or false."""
        if labels is None:
            labels = ["true", "false"]
        
        results = []
        
        # Keyword-based fallback patterns
        false_keywords = [
            "flat earth", "fake moon landing", "vaccines cause autism",
            "5g causes cancer", "covid hoax", "chemtrails", "lizard people"
        ]
        
        true_keywords = [
            "exercise is healthy", "smoking is harmful", "earth is round",
            "vaccines prevent disease", "climate change is real"
        ]
        
        for claim in claims:
            claim_lower = claim.lower()
            
            if self.pipeline:
                try:
                    result = self.pipeline(claim, labels)
                    prediction = result['labels'][0]
                    confidence = result['scores'][0]
                except Exception as e:
                    print(f"⚠️ Zero-shot classification failed: {e}")
                    # Use keyword fallback
                    prediction, confidence = self._keyword_classify(claim_lower, false_keywords, true_keywords)
            else:
                # Keyword-based classification
                prediction, confidence = self._keyword_classify(claim_lower, false_keywords, true_keywords)
            
            results.append({
                'prediction': prediction,
                'confidence': confidence,
                'claim': claim
            })
        
        return results
    
    def _keyword_classify(self, claim_lower, false_keywords, true_keywords):
        """Classify using keyword matching."""
        # Check for false claim indicators
        for keyword in false_keywords:
            if keyword in claim_lower:
                return "false", 0.85
        
        # Check for true claim indicators  
        for keyword in true_keywords:
            if keyword in claim_lower:
                return "true", 0.90
        
        # Default to uncertain
        return "false", 0.52


class FineTunedFactChecker:
    """Fine-tuned fact checking model."""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_data(self, dataset):
        """Prepare and tokenize the dataset."""
        if not transformers_available:
            print("⚠️ Transformers not available for fine-tuning")
            return None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            
            tokenized_dataset = {}
            for split_name, split_data in dataset.items():
                if datasets_available:
                    dataset_split = Dataset.from_dict({
                        'text': split_data['text'],
                        'labels': split_data['labels']
                    })
                    tokenized_dataset[split_name] = dataset_split.map(tokenize_function, batched=True)
                else:
                    print(f"⚠️ Cannot tokenize {split_name} split: datasets library not available")
            
            return tokenized_dataset
        
        except Exception as e:
            print(f"⚠️ Data preparation failed: {e}")
            return None
    
    def train(self, tokenized_dataset, output_dir="./fine_tuned_model"):
        """Train the fine-tuned model."""
        if not transformers_available or not tokenized_dataset:
            print("⚠️ Cannot train: missing dependencies or data")
            return
        
        try:
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            print("🚀 Starting fine-tuning...")
            self.trainer.train()
            print("✅ Fine-tuning completed!")
            
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
    
    def predict(self, texts):
        """Make predictions on new texts."""
        if not self.model or not self.tokenizer:
            print("⚠️ Model not trained or loaded")
            return []
        
        try:
            predictions = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred_class].item()
                
                predictions.append({
                    'prediction': pred_class,
                    'confidence': confidence,
                    'text': text
                })
            
            return predictions
        
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return []


class VerdictFormatter:
    """Formats model predictions into human-readable verdicts."""
    
    def format_verdicts(self, predictions):
        """Format predictions into readable verdicts."""
        formatted = []
        
        for pred in predictions:
            confidence = pred.get('confidence', 0.5)
            prediction = pred.get('prediction', 'unknown')
            
            # Determine verdict based on prediction and confidence
            if isinstance(prediction, str):
                if prediction.lower() == 'true' and confidence > 0.7:
                    verdict = "✅ TRUE - Claim appears factual"
                elif prediction.lower() == 'false' and confidence > 0.7:
                    verdict = "❌ FALSE - Claim appears to be misinformation" 
                else:
                    verdict = f"🤔 Uncertain NEEDS REVIEW (p={confidence:.2f})"
            else:
                # Numeric prediction (0=false, 1=true)
                if prediction == 1 and confidence > 0.7:
                    verdict = "✅ TRUE - Claim appears factual"
                elif prediction == 0 and confidence > 0.7:
                    verdict = "❌ FALSE - Claim appears to be misinformation"
                else:
                    verdict = f"🤔 Uncertain NEEDS REVIEW (p={confidence:.2f})"
            
            formatted.append({
                'verdict': verdict,
                'confidence': confidence,
                'raw_prediction': prediction,
                'text': pred.get('text', pred.get('claim', ''))
            })
        
        return formatted


class CounterNarrativeGenerator:
    """Generates counter-narratives for false claims."""
    
    def __init__(self, model_candidates=None):
        self.model_candidates = model_candidates or [
            "google/flan-t5-small",
            "t5-small"
        ]
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the best available text generation model."""
        print("🔧 Loading counter-narrative generator...")
        
        for model_name in self.model_candidates:
            try:
                print(f"🔧 Loading counter-narrative model: {model_name}")
                if transformers_available:
                    self.pipeline = pipeline(
                        "text2text-generation",
                        model=model_name,
                        tokenizer=model_name,
                        device=-1  # CPU only
                    )
                    print(f"✅ Successfully loaded: {model_name}")
                    return
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        print("   ⚠️ No counter-narrative models could be loaded (offline mode or connectivity issues)")
        print("   📝 Using simple template-based fallback")
    
    def generate_corrections(self, processed_dataset, formatted_verdicts, num_test_examples=3):
        """Generate counter-narratives for false claims."""
        print("🔄 Generating counter-narratives...")
        
        test_claims = processed_dataset['test']['text'][:num_test_examples]
        corrections = []
        
        for i, (claim, verdict) in enumerate(zip(test_claims, formatted_verdicts)):
            try:
                if "FALSE" in verdict['verdict'] or "Uncertain" in verdict['verdict']:
                    if self.pipeline:
                        # Try AI generation
                        prompt = f"Correct this misinformation with facts: {claim}"
                        result = self.pipeline(
                            prompt,
                            max_length=150,
                            min_length=50,
                            do_sample=True,
                            temperature=0.7,
                            truncation=True
                        )
                        correction = result[0]['generated_text']
                    else:
                        # Template-based fallback
                        correction = self._generate_template_correction(claim)
                else:
                    correction = "Not needed (claim is true)"
                
                corrections.append(correction)
                
            except Exception as e:
                print(f"⚠️ Counter-narrative generation failed for claim {i}: {e}")
                corrections.append(self._generate_template_correction(claim))
        
        return corrections
    
    def _generate_template_correction(self, claim):
        """Generate a template-based correction."""
        templates = [
            f"This claim requires fact-checking. Please verify with reliable sources before sharing.",
            f"According to scientific consensus, this statement needs verification from credible sources.",
            f"This appears to be misinformation. Please consult expert sources for accurate information."
        ]
        
        # Simple keyword-based template selection
        if any(word in claim.lower() for word in ['earth', 'flat', 'nasa']):
            return "Scientific evidence from multiple sources confirms the Earth is spherical, not flat. NASA and other space agencies provide extensive photographic and measurement evidence."
        elif any(word in claim.lower() for word in ['vaccine', 'autism']):
            return "Multiple large-scale scientific studies have found no link between vaccines and autism. Vaccines are safe and effective at preventing serious diseases."
        else:
            return templates[0]


# Configuration
DEFAULT_CONFIG = {
    'output_dir': './outputs',
    'model_cache_dir': './models',
    'device': 'cpu',  # Laptop-friendly
    'batch_size': 4,
    'max_length': 512,
    'num_epochs': 3,
    'learning_rate': 2e-5
}