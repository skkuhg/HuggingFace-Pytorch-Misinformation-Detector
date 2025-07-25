"""
Batch Processing Example for Misinformation Detector

This example demonstrates how to process multiple claims efficiently
and handle various scenarios like errors, different confidence levels, etc.
"""

from graphics_generator import MisinformationPipeline, PipelineEvaluator
import json
import os
from typing import List, Dict, Any
import time

def create_sample_dataset():
    """Create a sample dataset for batch processing."""
    
    return {
        'claims': [
            "The Earth is flat and governments are hiding this truth",
            "Vaccines cause autism and should be avoided",
            "Climate change is a hoax invented by scientists for funding",
            "5G cell towers cause cancer and COVID-19",
            "The moon landing was filmed in a Hollywood studio",
            "Regular exercise improves cardiovascular health",
            "Drinking water is essential for human survival",
            "Smoking cigarettes increases lung cancer risk",
            "The Great Wall of China is visible from space",
            "Eating fruits and vegetables provides essential nutrients"
        ],
        'expected_labels': [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # 0=false, 1=true
        'categories': [
            'conspiracy', 'health', 'science', 'technology', 'history',
            'health', 'basic_fact', 'health', 'geography', 'nutrition'
        ]
    }


def batch_processing_basic():
    """Basic batch processing example."""
    
    print("📦 Basic Batch Processing Example")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MisinformationPipeline()
    
    # Get sample data
    data = create_sample_dataset()
    
    # Process all claims
    start_time = time.time()
    results = pipeline.process_batch(
        claims=data['claims'],
        style='fact_check',
        save_dir='./batch_basic'
    )
    end_time = time.time()
    
    print(f"\n⏱️ Processing completed in {end_time - start_time:.2f} seconds")
    print(f"📊 Processed {len(results)} claims")
    
    # Analyze results
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    return results


def batch_with_categories():
    """Batch processing with category-based analysis."""
    
    print("\n🏷️ Batch Processing with Categories")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    data = create_sample_dataset()
    
    # Process claims and group by category
    results_by_category = {}
    
    for claim, category in zip(data['claims'], data['categories']):
        print(f"\nProcessing {category}: {claim[:50]}...")
        
        result = pipeline.process_claim(
            claim=claim,
            style='professional' if category in ['science', 'health'] else 'fact_check',
            save_dir=f'./batch_categories/{category}'
        )
        
        if category not in results_by_category:
            results_by_category[category] = []
        
        results_by_category[category].append({
            'claim': claim,
            'verdict': result['verdict']['verdict'],
            'confidence': result['verdict']['confidence']
        })
    
    # Print category summary
    print("\n📊 Results by Category:")
    for category, results in results_by_category.items():
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        false_claims = sum(1 for r in results if 'FALSE' in r['verdict'])
        
        print(f"\n{category.upper()}:")
        print(f"  Claims: {len(results)}")
        print(f"  False claims: {false_claims}")
        print(f"  Avg confidence: {avg_confidence:.3f}")


def batch_evaluation_example():
    """Example showing how to evaluate batch processing results."""
    
    print("\n📊 Batch Evaluation Example")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    data = create_sample_dataset()
    
    # Create test dataset format
    test_dataset = {
        'text': data['claims'],
        'labels': data['expected_labels']
    }
    
    # Evaluate the pipeline
    evaluator = PipelineEvaluator(pipeline)
    metrics = evaluator.evaluate_dataset(test_dataset)
    
    print("📈 Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    if metrics['errors']:
        print(f"\n❌ Errors encountered: {len(metrics['errors'])}")
        for error in metrics['errors'][:3]:  # Show first 3 errors
            print(f"  - Index {error['index']}: {error['error']}")
    
    return metrics


def batch_with_different_styles():
    """Process claims with different visual styles based on content."""
    
    print("\n🎨 Batch Processing with Style Selection")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    data = create_sample_dataset()
    
    # Define style mapping based on keywords
    style_rules = {
        'health': ['vaccine', 'cancer', 'disease', 'smoking', 'exercise'],
        'warning': ['hoax', 'conspiracy', 'hiding', 'fake'],
        'educational': ['nutrition', 'water', 'essential'],
        'professional': ['science', 'research', 'study']
    }
    
    def select_style(claim: str) -> str:
        """Select appropriate style based on claim content."""
        claim_lower = claim.lower()
        
        for style, keywords in style_rules.items():
            if any(keyword in claim_lower for keyword in keywords):
                return style
        
        return 'fact_check'  # default
    
    # Process with appropriate styles
    styled_results = []
    
    for i, claim in enumerate(data['claims']):
        style = select_style(claim)
        print(f"\nClaim {i+1} ({style}): {claim[:50]}...")
        
        result = pipeline.process_claim(
            claim=claim,
            style=style,
            save_dir=f'./batch_styled/{style}'
        )
        
        styled_results.append({
            'claim': claim,
            'style': style,
            'verdict': result['verdict']['verdict'],
            'save_path': result['save_path']
        })
    
    # Summary by style
    style_summary = {}
    for result in styled_results:
        style = result['style']
        if style not in style_summary:
            style_summary[style] = 0
        style_summary[style] += 1
    
    print("\n🎨 Style Usage Summary:")
    for style, count in style_summary.items():
        print(f"  {style}: {count} claims")


def save_batch_results(results: List[Dict[str, Any]], filename: str):
    """Save batch processing results to JSON file."""
    
    # Prepare results for JSON serialization
    json_results = []
    for result in results:
        json_result = {
            'original_claim': result.get('original_claim', ''),
            'summary': result.get('summary', ''),
            'verdict': result.get('verdict', {}).get('verdict', ''),
            'confidence': result.get('verdict', {}).get('confidence', 0.0),
            'correction': result.get('correction', ''),
            'style': result.get('style', ''),
            'save_path': result.get('save_path', ''),
            'status': 'success' if 'error' not in result else 'failed',
            'error': result.get('error', None)
        }
        json_results.append(json_result)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")


def main():
    """Run all batch processing examples."""
    
    print("🚀 Batch Processing Examples for Misinformation Detector")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('./batch_basic', exist_ok=True)
    os.makedirs('./batch_categories', exist_ok=True)
    os.makedirs('./batch_styled', exist_ok=True)
    
    try:
        # Run examples
        print("\n1. Basic Batch Processing")
        basic_results = batch_processing_basic()
        save_batch_results(basic_results, './batch_basic_results.json')
        
        print("\n2. Category-based Processing")
        batch_with_categories()
        
        print("\n3. Evaluation Example")
        metrics = batch_evaluation_example()
        
        print("\n4. Style-based Processing")
        batch_with_different_styles()
        
        print("\n🎉 All batch processing examples completed!")
        print("\nGenerated files:")
        print("  - ./batch_basic_results.json - Detailed results")
        print("  - ./batch_basic/ - Basic processing graphics")
        print("  - ./batch_categories/ - Category-organized graphics")
        print("  - ./batch_styled/ - Style-based graphics")
        
        return {
            'basic_results': basic_results,
            'evaluation_metrics': metrics
        }
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
        return None


if __name__ == "__main__":
    main()