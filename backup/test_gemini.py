import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_connection():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Try Gemini 1.5 Pro model directly
        model_name = "models/gemini-1.5-pro"
        logger.info(f"Attempting to use model: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello, this is a test message.")
            
            if response and response.text:
                logger.info("✅ Successfully connected to Gemini API")
                logger.info(f"Response: {response.text}")
                return True
            else:
                logger.error("Empty response from Gemini API")
                return False
                
        except Exception as e:
            logger.error(f"Failed to use {model_name}: {str(e)}")
            
            # If that fails, try listing models
            logger.info("\nListing available models...")
            models = genai.list_models()
            
            if not models:
                logger.error("No models returned from Gemini API")
                return False
                
            logger.info("=== Available Models ===")
            for model in models:
                logger.info(f"Model Name: {model.name}")
                logger.info(f"Display Name: {model.display_name}")
                logger.info(f"Generation Methods: {model.supported_generation_methods}")
                logger.info("---")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error connecting to Gemini API: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_connection()