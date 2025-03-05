from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import logging
import json
from openai import OpenAI
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  # For Claude
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # For Google Cloud Translation
logger.info(f"Loaded API Keys - OpenAI: {'Present' if OPENAI_API_KEY else 'Missing'}, Anthropic: {'Present' if ANTHROPIC_API_KEY else 'Missing'}, Google: {'Present' if GOOGLE_API_KEY else 'Missing'}")

# Setup API
app = FastAPI()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    CLAUDE = "claude-3"
    GOOGLE = "google-translate"

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create templates directory and HTML file
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Persian Text Editor and Translator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .text-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            height: 100px;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            resize: vertical;
        }
        .edited-text {
            width: 100%;
            min-height: 50px;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            background-color: #f8f8f8;
            text-align: right;
            direction: rtl;
        }
        .explanation {
            margin-top: 10px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 5px;
            font-size: 14px;
            color: #1976D2;
            text-align: left;
            direction: ltr;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            margin: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        button.edit {
            background-color: #2196F3;
        }
        button.edit:hover {
            background-color: #1976D2;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            min-height: 50px;
            background-color: white;
            font-size: 16px;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
        .error {
            color: #d32f2f;
            font-weight: bold;
        }
        select {
            padding: 8px;
            font-size: 16px;
            margin: 10px 0;
            border-radius: 5px;
            border: 2px solid #ddd;
        }
        .model-info {
            margin: 10px 0;
            font-size: 14px;
            color: #666;
        }
        .section-title {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <h1>Persian Text Editor and Translator</h1>
    <div class="container">
        <!-- Writing Section -->
        <div class="text-section">
            <div class="section-title">Write Persian Text</div>
            <textarea id="draft" placeholder="Write your Persian text here..." dir="rtl"></textarea>
            <button onclick="editText()" class="edit">Edit Text</button>
        </div>

        <!-- Editing Section -->
        <div class="text-section">
            <div class="section-title">Edit and Finalize</div>
            <div id="editedText" class="edited-text" contenteditable="true" dir="rtl"></div>
            <div id="explanation" class="explanation"></div>
            <button onclick="finalizeEdit()" class="edit">Finalize Edit</button>
        </div>

        <!-- Translation Section -->
        <div class="text-section">
            <div class="section-title">Translate to English</div>
            <div class="model-selection">
                <select id="model" onchange="updateModelInfo()">
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Good)</option>
                    <option value="gpt-4">GPT-4 (Most Accurate)</option>
                    <option value="claude-3">Claude-3 (Accurate)</option>
                    <option value="google-translate">Google Translate</option>
                </select>
                <div id="modelInfo" class="model-info"></div>
            </div>
            <div id="finalText" class="edited-text" dir="rtl"></div>
            <button onclick="translateText()">Translate</button>
            <div id="result"></div>
        </div>
    </div>

    <script>
        // Add API keys to JavaScript
        const OPENAI_API_KEY = '{{ OPENAI_API_KEY }}';
        const ANTHROPIC_API_KEY = '{{ ANTHROPIC_API_KEY }}';
        const GOOGLE_API_KEY = '{{ GOOGLE_API_KEY }}';

        async function translate_with_openai(text, model) {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: model
                })
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Translation failed');
            }
            
            const data = await response.json();
            return data.translation;
        }

        async function translate_with_claude(text) {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: 'claude-3'
                })
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Translation failed');
            }
            
            const data = await response.json();
            return data.translation;
        }

        async function translate_with_gemini(text) {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: 'google-translate'
                })
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Translation failed');
            }
            
            const data = await response.json();
            return data.translation;
        }

        function updateModelInfo() {
            const model = document.getElementById('model').value;
            const modelInfo = document.getElementById('modelInfo');
            const modelDescriptions = {
                'gpt-3.5-turbo': 'Fast and reliable translations with good accuracy',
                'gpt-4': 'Most accurate translations, better understanding of context and nuances',
                'claude-3': 'Accurate translations',
                'google-translate': 'Accurate translations'
            };
            modelInfo.textContent = modelDescriptions[model];
        }

        async function editText() {
            const draftText = document.getElementById('draft').value;
            const editedText = document.getElementById('editedText');
            const explanation = document.getElementById('explanation');
            
            if (!draftText.trim()) {
                editedText.innerHTML = '<span class="error">Please enter some text to edit</span>';
                return;
            }
            
            editedText.innerHTML = '<span class="loading">AI is improving the text...</span>';
            explanation.innerHTML = '';
            
            try {
                const response = await fetch('/edit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: draftText })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Edit failed');
                }
                
                const [improvedText, explanationText] = data.improved_text.split('|').map(s => s.trim());
                editedText.textContent = improvedText;
                explanation.textContent = explanationText;
            } catch (error) {
                editedText.innerHTML = `<span class="error">❌ Error: ${error.message}</span>`;
                console.error('Edit error:', error);
            }
        }

        function finalizeEdit() {
            const editedText = document.getElementById('editedText').textContent;
            document.getElementById('finalText').textContent = editedText;
        }

        async function translateText() {
            const text = document.getElementById('finalText').textContent;
            const model = document.getElementById('model').value;
            const result = document.getElementById('result');
            
            if (!text.trim()) {
                result.innerHTML = '<span class="error">Please enter some text to translate</span>';
                return;
            }
            
            result.innerHTML = '<span class="loading">Translating...</span>';
            
            try {
                let translation;
                if (model === 'gpt-3.5-turbo' || model === 'gpt-4') {
                    translation = await translate_with_openai(text, model);
                } else if (model === 'claude-3') {
                    translation = await translate_with_claude(text);
                } else if (model === 'google-translate') {
                    translation = await translate_with_gemini(text);
                }
                result.innerHTML = translation;
            } catch (error) {
                result.innerHTML = `<span class="error">❌ Error: ${error.message}</span>`;
                console.error('Translation error:', error);
            }
        }

        // Initialize model info
        updateModelInfo();

        // Add keyboard shortcuts
        document.getElementById('draft').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                editText();
            }
        });

        document.getElementById('editedText').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                finalizeEdit();
            }
        });
    </script>
</body>
</html>
    """)

class EditRequest(BaseModel):
    text: str

class TranslationRequest(BaseModel):
    text: str
    model: ModelType

async def translate_with_openai(text: str, model: str) -> str:
    try:
        logger.info(f"Starting OpenAI translation with model {model}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Persian to English translator. Translate the following text accurately and naturally."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        translation = response.choices[0].message.content.strip()
        logger.info(f"OpenAI translation successful: {translation}")
        return translation
    except Exception as e:
        logger.error(f"OpenAI translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI translation failed: {str(e)}")

async def translate_with_claude(text: str) -> str:
    try:
        logger.info("Starting Claude translation")
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "content-type": "application/json",
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate this Persian text to English: {text}"
                    }
                ]
            }
        )
        response.raise_for_status()
        translation = response.json()["content"][0]["text"]
        logger.info(f"Claude translation successful: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Claude translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Claude translation failed: {str(e)}")

async def translate_with_gemini(text: str) -> str:
    try:
        logger.info("Starting Google Translate translation")
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            'q': text,
            'target': 'en',
            'source': 'fa',
            'key': GOOGLE_API_KEY
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        result = response.json()
        translation = result['data']['translations'][0]['translatedText'].strip()
        logger.info(f"Google Translate successful: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Google Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google Translation failed: {str(e)}")

async def edit_with_openai(text: str) -> str:
    system_prompt = """You are an expert Persian language editor specializing in Persian poetry and literary texts. 
    You have deep knowledge of:
    1. Classical Persian poetry (شعر کلاسیک فارسی)
    2. Persian grammar and syntax (دستور زبان فارسی)
    3. Persian literary devices (صنایع ادبی)
    4. Common Persian expressions and idioms (اصطلاحات و ضرب‌المثل‌ها)
    5. Persian orthography and spelling (املای فارسی)
    6. Persian punctuation (نشانه‌گذاری)
    7. Persian poetic meters (وزن شعر)
    8. Persian literary style (سبک ادبی)

    Your task is to:
    1. Correct any spelling or orthography errors
    2. Fix grammar and syntax issues
    3. Maintain the poetic/literary style
    4. Preserve the original meaning and rhythm
    5. Ensure proper punctuation
    6. Keep the formal literary tone

    For the text: "{text}"
    
    Provide the improved text with a detailed explanation of changes, especially for:
    - Spelling corrections
    - Grammar fixes
    - Punctuation adjustments
    - Style improvements
    
    Format: [Improved Text] | [Detailed Explanation]"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3,  # Lower temperature for more precise corrections
        max_tokens=1000,  # Ensure enough tokens for detailed explanation
        presence_penalty=0.1  # Slight penalty to encourage focused corrections
    )
    return response.choices[0].message.content.strip()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        logger.info(f"Attempting to translate text using {request.model}: {request.text}")
        
        if request.model in [ModelType.GPT35, ModelType.GPT4]:
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key is missing")
            translation = await translate_with_openai(request.text, request.model)
        
        elif request.model == ModelType.CLAUDE:
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key is missing")
            translation = await translate_with_claude(request.text)
        
        elif request.model == ModelType.GOOGLE:
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key is missing")
            translation = await translate_with_gemini(request.text)
        
        logger.info(f"Translation successful: {translation}")
        return {"translation": translation}
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit")
async def edit(request: EditRequest):
    try:
        logger.info(f"Attempting to edit text: {request.text}")
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key is missing")

        improved_text = await edit_with_openai(request.text)
        logger.info(f"Edit successful: {improved_text}")
        
        return {"improved_text": improved_text}
            
    except Exception as e:
        logger.error(f"Edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)