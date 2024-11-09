import os
from pathlib import Path
from openai import OpenAI
import base64
from PIL import Image
import json
from datetime import datetime
import logging

class PhotoPromptGenerator:
    def __init__(self, api_key, input_folder, output_folder):
        """
        Initialize the generator with API key and folder paths.
        
        Args:
            api_key (str): OpenAI API key
            input_folder (str): Path to folder containing photos
            output_folder (str): Path to save generated prompts
        """
        self.client = OpenAI(api_key=api_key)
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prompt_generator.log'),
                logging.StreamHandler()
            ]
        )
        
    def encode_image(self, image_path):
        """
        Encode image to base64 string.
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_image_prompt(self, image_path):
        """
        Generate prompt for image using GPT-4 Vision.
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            str: Generated Runway prompt
        """
        base64_image = self.encode_image(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing film photos and creating prompts for Runway AI to generate subtle, aesthetic videos. 
                        Format your response as a detailed prompt that maintains the film aesthetic while adding gentle motion.
                        Include specifications for: frame rate (24fps), motion elements, static elements, style preservation (film grain, color temperature),
                        composition requirements, and loop duration (5 seconds). Focus on subtle, natural movements. Please format the output as a simple text 
                        paragraph in less than 512 characters."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Create a Runway AI prompt for this film photo that will create a subtle, living cinemagraph while maintaining the original film aesthetic."
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating prompt for {image_path}: {str(e)}")
            return None

    def save_prompt(self, image_name, prompt):
        """
        Save generated prompt to output folder.
        
        Args:
            image_name (str): Original image filename
            prompt (str): Generated prompt
        """
        output_file = self.output_folder / f"{image_name}_prompt.txt"
        try:
            with open(output_file, 'w') as f:
                f.write(prompt)
            logging.info(f"Saved prompt for {image_name}")
        except Exception as e:
            logging.error(f"Error saving prompt for {image_name}: {str(e)}")

    def process_photos(self):
        """Process all photos in input folder and generate prompts"""
        self.output_folder.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        for image_path in self.input_folder.iterdir():
            if image_path.suffix.lower() in image_extensions:
                logging.info(f"Processing {image_path.name}")
                
                prompt = self.get_image_prompt(image_path)
                if prompt:
                    self.save_prompt(image_path.stem, prompt)

def main():
    # Replace with your actual API key and folder paths
    api_key = os.getenv('OPENAI_API_KEY')
    input_folder = "input-photos"
    output_folder = "prompts"
    
    generator = PhotoPromptGenerator(api_key, input_folder, output_folder)
    generator.process_photos()

if __name__ == "__main__":
    main()