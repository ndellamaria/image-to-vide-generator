import os
from pathlib import Path
import time
import requests
from runwayml import RunwayML
import logging
import base64

class SimpleVideoGenerator:
    def __init__(self, input_prompt_folder: str, output_video_folder: str):
        """
        Initialize the Runway video generator with simplified client.
        
        Args:
            input_prompt_folder (str): Folder containing generated prompts
            output_video_folder (str): Folder to save generated videos
        """
        self.runway_key = os.getenv('RUNWAY_API_KEY')
        if not self.runway_key:
            raise ValueError("RUNWAY_API_KEY not found in environment variables")
            
        self.client = RunwayML(api_key=self.runway_key)
        self.input_folder = Path(input_prompt_folder)
        self.output_folder = Path(output_video_folder)
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('runway_generator.log'),
                logging.StreamHandler()
            ]
        )

    def download_video(self, url: str, output_path: Path) -> bool:
        """
        Download video from URL to specified path.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        
            logging.info(f"Successfully downloaded video to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error downloading video: {str(e)}")
            return False
        
    def generate_video(self, prompt_path: Path):
        """
        Generate a video from a single prompt file.
        
        Args:
            prompt_path (Path): Path to the prompt file
        """
        try:
            # Read prompt
            with open(prompt_path, 'r') as f:
                prompt = f.read()

            # Find corresponding image
            image_name = prompt_path.stem.replace('_prompt', '')
            image_path = self.input_folder.parent / 'input-photos' / f"{image_name}.jpg"
            
            if not image_path.exists():
                logging.error(f"Could not find image for {prompt_path}")
                return

            # Create the generation task
            logging.info(f"Creating video generation task for {image_name}")

            
            # Convert image to base64 string
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            task = self.client.image_to_video.create(
                model='gen3a_turbo',
                prompt_image=f"data:image/png;base64,{base64_image}",
                prompt_text=prompt
            )

            task_id = task.id
            time.sleep(10)
            task = self.client.tasks.retrieve(task_id)

            # Poll for completion
            logging.info(f"Waiting for task {task.id} to complete")
            while task.status not in ['SUCCEEDED', 'FAILED']:
                time.sleep(10)  # Wait for 10 seconds before polling
                task = self.client.tasks.retrieve(task.id)
                logging.info(f"Task status: {task.status}")

            if task.status == 'SUCCEEDED':
                # Get the video URL from the output
                if task.output is not None and len(task.output) > 0:
                    video_url = task.output[0]
                    logging.info(f"Video URL: {video_url}")
                else:
                    logging.error(f"No output url for {image_name}")
                
                if video_url:
                    output_path = self.output_folder / f"{image_name}.mp4"
                    if self.download_video(video_url, output_path):
                        logging.info(f"Video generation complete for {image_name}")
                    else:
                        logging.error(f"Failed to download video for {image_name}")
                else:
                    logging.error(f"No video URL in task output for {image_name}")
            else:
                logging.error(f"Task failed for {image_name}")

        except Exception as e:
            logging.error(f"Error processing {prompt_path}: {str(e)}")

    def process_all_prompts(self):
        """Process all prompt files and generate videos"""
        self.output_folder.mkdir(exist_ok=True)
        
        for prompt_path in self.input_folder.glob('*_prompt.txt'):
            logging.info(f"Processing {prompt_path.name}")
            self.generate_video(prompt_path)

def main():
    # Specify your folder paths
    input_prompt_folder = "prompts"
    output_video_folder = "videos"
    
    try:
        generator = SimpleVideoGenerator(input_prompt_folder, output_video_folder)
        generator.process_all_prompts()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()