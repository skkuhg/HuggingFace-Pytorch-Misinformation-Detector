"""
Basic Usage Examples for Misinformation Detector

This file demonstrates basic usage of the misinformation detection pipeline.
"""

from graphics_generator import MisinformationPipeline
import os

def basic_example():
    """Basic single claim processing example."""
    
    print("🚀 Basic Example: Single Claim Processing")
    print("=" * 50)
    
    # Initialize the pipeline
    pipeline = MisinformationPipeline()
    
    # Test claim
    claim = "The Earth is flat and NASA is lying to us"
    
    # Process the claim
    result = pipeline.process_claim(
        claim=claim,
        style="fact_check",
        save_dir="./example_outputs"
    )
    
    # Display results
    print(f"Original Claim: {result['original_claim']}")
    print(f"Summary: {result['summary']}")
    print(f"Verdict: {result['verdict']['verdict']}")
    print(f"Correction: {result['correction'][:100]}...")
    
    if result['save_path']:
        print(f"Graphic saved to: {result['save_path']}")
    
    return result


def multiple_styles_example():
    """Example showing different visual styles."""
    
    print("\n🎨 Multiple Styles Example")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    claim = "5G towers cause cancer and should be banned"
    
    styles = ['fact_check', 'warning', 'professional', 'educational']
    
    for style in styles:
        print(f"\nProcessing with style: {style}")
        result = pipeline.process_claim(
            claim=claim,
            style=style,
            save_dir=f"./style_examples/{style}"
        )
        print(f"✅ Generated {style} graphic")


def health_claims_example():
    """Example focused on health-related misinformation."""
    
    print("\n🏥 Health Claims Example")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    
    health_claims = [
        "Vaccines cause autism in children",
        "COVID-19 is just a common cold",
        "Drinking bleach can cure coronavirus",
        "Regular exercise reduces risk of heart disease",
        "Smoking has no health risks"
    ]
    
    for i, claim in enumerate(health_claims, 1):
        print(f"\n--- Processing Health Claim {i} ---")
        result = pipeline.process_claim(
            claim=claim,
            style="warning" if "vaccine" in claim.lower() or "bleach" in claim.lower() else "professional",
            save_dir="./health_claims"
        )
        
        print(f"Claim: {claim}")
        print(f"Verdict: {result['verdict']['verdict']}")


def confidence_analysis_example():
    """Example showing confidence analysis."""
    
    print("\n📊 Confidence Analysis Example")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    
    test_claims = [
        "The sky is blue",  # Obviously true
        "The Earth is flat",  # Obviously false
        "Artificial intelligence will replace all jobs by 2030",  # Uncertain
        "Water boils at 100°C at sea level",  # Scientific fact
        "Aliens built the pyramids"  # Conspiracy theory
    ]
    
    results = []
    for claim in test_claims:
        result = pipeline.process_claim(claim, save_dir="./confidence_analysis")
        results.append({
            'claim': claim,
            'confidence': result['verdict']['confidence'],
            'verdict': result['verdict']['verdict']
        })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("\nResults sorted by confidence:")
    for result in results:
        print(f"Confidence: {result['confidence']:.3f} | {result['verdict']}")
        print(f"Claim: {result['claim']}\n")


def batch_processing_simple():
    """Simple batch processing example."""
    
    print("\n📦 Simple Batch Processing")
    print("=" * 50)
    
    pipeline = MisinformationPipeline()
    
    claims = [
        "Climate change is a hoax created by scientists",
        "Regular handwashing prevents the spread of germs",
        "The moon landing was filmed in a Hollywood studio"
    ]
    
    # Process all claims at once
    results = pipeline.process_batch(
        claims=claims,
        style="fact_check",
        save_dir="./batch_simple"
    )
    
    print(f"Processed {len(results)} claims successfully!")
    
    for i, result in enumerate(results, 1):
        if 'error' not in result:
            print(f"\nClaim {i}: {result['verdict']['verdict']}")
        else:
            print(f"\nClaim {i}: Failed - {result['error']}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs("./example_outputs", exist_ok=True)
    
    # Run all examples
    try:
        basic_example()
        multiple_styles_example()
        health_claims_example()
        confidence_analysis_example()
        batch_processing_simple()
        
        print("\n🎉 All examples completed successfully!")
        print("Check the generated directories for output files.")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")