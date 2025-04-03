import os
import sys
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_vector_embedding import LLMCaptionGenerator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a caption for an image using LLM')
    parser.add_argument('image_url', help='URL of the image to caption')
    parser.add_argument('--model', default='o1', help='OpenAI model to use (default: o3-mini)')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Create caption generator
    try:
        caption_generator = LLMCaptionGenerator(model=args.model)
        print(f"Using model: {args.model}")
        print(f"Generating caption for image: {args.image_url}")
        
        # Generate caption
        caption = caption_generator.generate_caption(args.image_url)
        
        # Print the result
        print("\n--- Generated Caption ---")
        print(caption)
        print("------------------------")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 