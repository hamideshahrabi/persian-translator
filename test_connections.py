import os
from dotenv import load_dotenv
import logging
import openai
import google.generativeai as genai
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OpenAI API key not found")
            return False
            
        logger.info(f"Testing OpenAI connection with API key (masked): {api_key[:8]}...")
        
        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        if response and response.choices:
            logger.info("✅ OpenAI connection successful")
            return True
        else:
            logger.error("❌ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ OpenAI connection error: {str(e)}")
        return False

async def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("Gemini API key not found")
            return False
            
        logger.info(f"Testing Gemini connection with API key (masked): {api_key[:8]}...")
        
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        logger.info("Available Gemini models:")
        for m in models:
            logger.info(f"Model: {m.name}")
            logger.info(f"Display name: {m.display_name}")
            logger.info(f"Description: {m.description}")
            logger.info(f"Generation methods: {m.supported_generation_methods}")
            logger.info("---")
        
        # Try to use gemini-1.5-flash-8b model
        model = genai.GenerativeModel('models/gemini-1.5-flash-8b')
        response = model.generate_content("Test message")
        
        if response and response.text:
            logger.info("✅ Gemini connection successful")
            return True
        else:
            logger.error("❌ Gemini API returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini connection error: {str(e)}")
        return False

async def main():
    """Run all connection tests"""
    logger.info("Starting API connection tests...")
    
    # Test OpenAI
    openai_success = await test_openai_connection()
    
    # Test Gemini
    gemini_success = await test_gemini_connection()
    
    # Print summary
    logger.info("\n=== Test Results ===")
    logger.info(f"OpenAI: {'✅ Connected' if openai_success else '❌ Failed'}")
    logger.info(f"Gemini: {'✅ Connected' if gemini_success else '❌ Failed'}")
    logger.info("==================")

if __name__ == "__main__":
    asyncio.run(main()) 