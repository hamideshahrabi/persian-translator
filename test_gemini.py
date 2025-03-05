from google.cloud import translate_v2 as translate
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize the translation client
translate_client = translate.Client()

# Test translation
persian_text = "سلام، چطور هستید؟"

try:
    # Translate the text
    result = translate_client.translate(
        persian_text,
        target_language='en',
        source_language='fa'
    )
    
    print(f"Original Persian text: {persian_text}")
    print(f"Translation: {result['translatedText']}")
except Exception as e:
    print(f"Error: {str(e)}")