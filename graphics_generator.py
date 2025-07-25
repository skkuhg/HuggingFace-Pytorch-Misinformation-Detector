"""
Graphics Generation Module for Misinformation Detection

This module handles the creation of "Myth vs Fact" visual graphics to counter
misinformation with clear, professional-looking infographics.

Components:
- VisualPromptBuilder: Creates prompts for image generation
- GraphicsGenerator: Generates myth vs fact graphics
- MisinformationPipeline: Complete end-to-end pipeline
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Try importing diffusers with fallback
try:
    from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
    diffusers_available = True
except ImportError:
    diffusers_available = False
    print("⚠️ Diffusers not available. Install with: pip install diffusers")

# Import core detection module
from misinformation_detector import (
    ClaimSummarizer, ZeroShotFactChecker, FineTunedFactChecker,
    VerdictFormatter, CounterNarrativeGenerator, DEFAULT_CONFIG
)


class VisualPromptBuilder:
    """Builds prompts for visual content generation."""
    
    def __init__(self):
        self.style_templates = {
            'fact_check': {
                'base_prompt': 'fact-checking infographic, split layout, comparison design, truth vs myth theme',
                'style': 'split screen design, left side labeled "MYTH" in red, right side labeled "FACT" in green',
                'quality': 'clean background, professional layout, no text content in the image, placeholder text areas',
                'technical': 'high quality, 16:9 aspect ratio, minimalist design'
            },
            'professional': {
                'base_prompt': 'professional educational infographic, corporate style, clean design',
                'style': 'blue and white color scheme, modern typography, structured layout',
                'quality': 'high-end design, minimal text, focus on visual hierarchy',
                'technical': 'professional quality, corporate aesthetics, 16:9 format'
            },
            'social_media': {
                'base_prompt': 'social media infographic, eye-catching design, viral content style',
                'style': 'bright colors, bold contrasts, attention-grabbing layout',
                'quality': 'shareable format, mobile-friendly, clear visual messaging',
                'technical': 'social media optimized, square format, high engagement design'
            },
            'educational': {
                'base_prompt': 'educational poster, academic style, informative design',
                'style': 'neutral colors, clear typography, structured information layout',
                'quality': 'classroom appropriate, fact-focused, evidence-based design',
                'technical': 'educational quality, print-ready, accessible design'
            },
            'warning': {
                'base_prompt': 'warning infographic, alert design, important notice style',
                'style': 'red and yellow warning colors, attention symbols, urgent messaging',
                'quality': 'clear warnings, high visibility, important information emphasis',
                'technical': 'warning format, high contrast, emergency communication style'
            }
        }
    
    def build_prompt(self, myth_text: str, fact_text: str, style: str = 'fact_check') -> Dict[str, str]:
        """Build a complete prompt for image generation."""
        template = self.style_templates.get(style, self.style_templates['fact_check'])
        
        # Create the main diffusion prompt
        diffusion_prompt = f"{template['base_prompt']}, {template['style']},\n        {template['quality']},\n        {template['technical']}"
        
        # Wrap text for better display
        myth_wrapped = self._wrap_text(myth_text, 25)
        fact_wrapped = self._wrap_text(fact_text, 25)
        
        return {
            'diffusion_prompt': diffusion_prompt,
            'myth_text': myth_text,
            'fact_text': fact_text,
            'myth_wrapped': myth_wrapped,
            'fact_wrapped': fact_wrapped,
            'style': style
        }
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        if len(text) <= width:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)


class GraphicsGenerator:
    """Generates myth vs fact graphics using AI image generation or fallback methods."""
    
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_name = model_name
        self.pipeline = None
        self.visual_builder = VisualPromptBuilder()
        self.load_model()
    
    def load_model(self):
        """Load the image generation model."""
        print("🎨 Loading graphics generator...")
        
        if not diffusers_available:
            print("   ⚠️ Diffusers not available, using fallback graphics")
            return
        
        try:
            print(f"🔧 Loading Stable Diffusion XL: {self.model_name}")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            
            # Move to appropriate device
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                print("✅ SDXL loaded on GPU")
            else:
                print("✅ SDXL loaded on CPU (may be slow)")
                
        except Exception as e:
            print(f"   ❌ Failed to load SDXL: {str(e)[:100]}...")
            print("   📝 Using simple graphics fallback")
            self.pipeline = None
    
    def generate_graphic(self, claim: str, correction: str, style: str = 'fact_check', 
                        save_path: Optional[str] = None) -> Image.Image:
        """Generate a myth vs fact graphic."""
        
        # Build the visual prompt
        prompt_data = self.visual_builder.build_prompt(claim, correction, style)
        
        if self.pipeline:
            try:
                print(f"🎨 Generating graphic with style: {style}")
                
                # Generate the image
                image = self.pipeline(
                    prompt=prompt_data['diffusion_prompt'],
                    negative_prompt="text, words, letters, watermark, signature, blurry, low quality",
                    height=576,
                    width=1024,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
                # Add text overlay
                image = self._add_text_overlay(image, prompt_data)
                
            except Exception as e:
                print(f"⚠️ AI generation failed: {e}")
                image = self._create_fallback_graphic(prompt_data)
        else:
            # Create fallback graphic
            image = self._create_fallback_graphic(prompt_data)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
            print(f"💾 Graphic saved to: {save_path}")
        
        return image
    
    def _add_text_overlay(self, image: Image.Image, prompt_data: Dict[str, str]) -> Image.Image:
        """Add text overlay to the generated image."""
        try:
            # Create a copy to draw on
            img_with_text = image.copy()
            draw = ImageDraw.Draw(img_with_text)
            
            # Try to load a font
            try:
                font_large = ImageFont.truetype("arial.ttf", 32)
                font_medium = ImageFont.truetype("arial.ttf", 24)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            
            # Image dimensions
            width, height = img_with_text.size
            
            # Draw "MYTH" and "FACT" labels
            draw.text((width//4, 50), "MYTH", fill='red', font=font_large, anchor='mm')
            draw.text((3*width//4, 50), "FACT", fill='green', font=font_large, anchor='mm')
            
            # Draw myth text (left side)
            myth_lines = prompt_data['myth_wrapped'].split('\n')
            y_start = height//3
            for i, line in enumerate(myth_lines[:4]):  # Max 4 lines
                draw.text((width//4, y_start + i*30), line, fill='black', font=font_medium, anchor='mm')
            
            # Draw fact text (right side)
            fact_lines = prompt_data['fact_wrapped'].split('\n')
            for i, line in enumerate(fact_lines[:4]):  # Max 4 lines
                draw.text((3*width//4, y_start + i*30), line, fill='black', font=font_medium, anchor='mm')
            
            return img_with_text
            
        except Exception as e:
            print(f"⚠️ Text overlay failed: {e}")
            return image
    
    def _create_fallback_graphic(self, prompt_data: Dict[str, str]) -> Image.Image:
        """Create a simple fallback graphic when AI generation fails."""
        print("🎨 Creating fallback graphic...")
        
        # Create a simple split-screen design
        width, height = 1024, 576
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to load fonts
        try:
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_text = ImageFont.truetype("arial.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # Draw split background
        draw.rectangle([0, 0, width//2, height], fill='#ffebee')  # Light red for myth
        draw.rectangle([width//2, 0, width, height], fill='#e8f5e8')  # Light green for fact
        
        # Draw center divider
        draw.line([width//2, 0, width//2, height], fill='gray', width=3)
        
        # Draw titles
        draw.text((width//4, 60), "MYTH", fill='red', font=font_title, anchor='mm')
        draw.text((3*width//4, 60), "FACT", fill='green', font=font_title, anchor='mm')
        
        # Draw content text
        myth_lines = prompt_data['myth_wrapped'].split('\n')
        fact_lines = prompt_data['fact_wrapped'].split('\n')
        
        y_start = 150
        line_height = 25
        
        # Draw myth text
        for i, line in enumerate(myth_lines[:8]):  # Max 8 lines
            draw.text((width//4, y_start + i*line_height), line, fill='black', font=font_text, anchor='mm')
        
        # Draw fact text
        for i, line in enumerate(fact_lines[:8]):  # Max 8 lines
            draw.text((3*width//4, y_start + i*line_height), line, fill='black', font=font_text, anchor='mm')
        
        return image


class MisinformationPipeline:
    """Complete end-to-end misinformation detection and counter-graphics pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        
        # Initialize components
        self.summarizer = ClaimSummarizer()
        self.zero_shot_checker = ZeroShotFactChecker()
        self.fine_tuned_checker = None  # Initialize when needed
        self.verdict_formatter = VerdictFormatter()
        self.counter_generator = CounterNarrativeGenerator()
        self.graphics_generator = GraphicsGenerator()
        
        print("✅ Misinformation detection pipeline initialized!")
    
    def process_claim(self, claim: str, style: str = 'fact_check', 
                     save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a single claim through the complete pipeline."""
        
        print(f"\n🔍 Processing claim: {claim[:100]}...")
        
        # Step 1: Summarize if needed
        if len(claim) > 200:
            summary = self.summarizer.summarize_claims([claim])[0]
            print(f"📝 Summarized: {summary}")
        else:
            summary = claim
        
        # Step 2: Fact check
        zero_shot_result = self.zero_shot_checker.check_claims([summary])[0]
        print(f"🔍 Zero-shot result: {zero_shot_result['prediction']} (confidence: {zero_shot_result['confidence']:.2f})")
        
        # Step 3: Format verdict
        verdict = self.verdict_formatter.format_verdicts([zero_shot_result])[0]
        print(f"⚖️ Verdict: {verdict['verdict']}")
        
        # Step 4: Generate counter-narrative if needed
        if "FALSE" in verdict['verdict'] or "Uncertain" in verdict['verdict']:
            correction = self.counter_generator._generate_template_correction(claim)
            print(f"📝 Generated correction: {correction[:100]}...")
        else:
            correction = "Claim appears to be accurate. No correction needed."
        
        # Step 5: Generate graphic
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"claim_graphic_{hash(claim) % 10000}.png")
        else:
            save_path = None
        
        graphic = self.graphics_generator.generate_graphic(
            claim=summary,
            correction=correction,
            style=style,
            save_path=save_path
        )
        
        # Return complete result
        result = {
            'original_claim': claim,
            'summary': summary,
            'verdict': verdict,
            'correction': correction,
            'graphic': graphic,
            'save_path': save_path,
            'style': style
        }
        
        print("✅ Claim processing complete!")
        return result
    
    def process_batch(self, claims: List[str], style: str = 'fact_check',
                     save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple claims in batch."""
        
        print(f"\n📊 Processing batch of {len(claims)} claims...")
        results = []
        
        for i, claim in enumerate(claims):
            print(f"\n--- Processing claim {i+1}/{len(claims)} ---")
            try:
                result = self.process_claim(claim, style, save_dir)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to process claim {i+1}: {e}")
                results.append({
                    'original_claim': claim,
                    'error': str(e),
                    'status': 'failed'
                })
        
        print(f"\n✅ Batch processing complete! {len(results)} results generated.")
        return results


# Evaluation utilities
class PipelineEvaluator:
    """Evaluates the performance of the misinformation detection pipeline."""
    
    def __init__(self, pipeline: MisinformationPipeline):
        self.pipeline = pipeline
    
    def evaluate_dataset(self, test_dataset: Dict) -> Dict[str, Any]:
        """Evaluate pipeline performance on a test dataset."""
        
        print("📊 Starting pipeline evaluation...")
        
        # Get predictions
        predictions = []
        true_labels = []
        errors = []
        
        for i, (text, label) in enumerate(zip(test_dataset['text'], test_dataset['labels'])):
            try:
                result = self.pipeline.zero_shot_checker.check_claims([text])[0]
                pred_label = 1 if result['prediction'] == 'true' else 0
                predictions.append(pred_label)
                true_labels.append(label)
            except Exception as e:
                errors.append({'index': i, 'error': str(e), 'text': text})
                predictions.append(0)  # Default to false
                true_labels.append(label)
        
        # Calculate metrics
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Create confusion matrix
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
        
        confusion_matrix = [[tn, fp], [fn, tp]]
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix,
            'errors': errors,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        print(f"📊 Evaluation complete! Accuracy: {accuracy:.3f}")
        return results


# Demo function
def run_demo():
    """Run a simple demonstration of the pipeline."""
    
    print("🚀 Starting Misinformation Detection Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MisinformationPipeline()
    
    # Demo claims
    demo_claims = [
        "The Earth is flat and NASA is lying to us",
        "Vaccines cause autism in children",
        "Regular exercise is beneficial for cardiovascular health"
    ]
    
    # Process each claim
    results = []
    for claim in demo_claims:
        result = pipeline.process_claim(
            claim=claim,
            style='fact_check',
            save_dir='./demo_outputs'
        )
        results.append(result)
    
    print("\n🎉 Demo completed!")
    print("Check the './demo_outputs' directory for generated graphics.")
    
    return results


if __name__ == "__main__":
    # Run demo if executed directly
    import torch
    run_demo()