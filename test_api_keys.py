import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Test OpenAI API
def test_openai():
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("OpenAI API test successful!")
        return True
    except Exception as e:
        print(f"OpenAI API test failed: {str(e)}")
        return False

# Test Gemini API
def test_gemini():
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content("Hello")
        print("Gemini API test successful!")
        return True
    except Exception as e:
        print(f"Gemini API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing API keys...")
    openai_result = test_openai()
    gemini_result = test_gemini()
    
    if openai_result and gemini_result:
        print("\nAll API keys are working correctly!")
    else:
        print("\nSome API keys failed the test. Please check the errors above.") 