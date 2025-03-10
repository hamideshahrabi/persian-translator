from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import logging
import json
from enum import Enum
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  # For Claude
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # For Google Cloud Translation
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Separate key for Gemini
logger.info(f"Loaded API Keys - OpenAI: {'Present' if OPENAI_API_KEY else 'Missing'}, "
           f"Anthropic: {'Present' if ANTHROPIC_API_KEY else 'Missing'}, "
           f"Google Cloud: {'Present' if GOOGLE_API_KEY else 'Missing'}, "
           f"Gemini: {'Present' if GEMINI_API_KEY else 'Missing'}")

# Setup API
app = FastAPI()

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    CLAUDE = "claude-3"
    GOOGLE = "google-translate"
    GEMINI = "gemini"  # Adding Gemini as new option

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create templates directory and HTML file
os.makedirs("templates", exist_ok=True)
with open("templates/index.html", "w") as f:
    f.write('''
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
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .text-section:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        textarea {
            width: 100%;
            height: 120px;
            margin: 10px 0;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        textarea:focus {
            border-color: #2196F3;
            outline: none;
        }
        .edited-text {
            width: 100%;
            min-height: 50px;
            margin: 15px 0;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background-color: #f8f9fa;
            text-align: right;
            direction: rtl;
        }
        .explanation {
            margin-top: 15px;
            padding: 12px;
            background-color: #e3f2fd;
            border-radius: 8px;
            font-size: 14px;
            color: #1976D2;
            text-align: left;
            direction: ltr;
            border-left: 4px solid #1976D2;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 8px;
        }
        button:hover {
            background-color: #1976D2;
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(1px);
        }
        .loading {
            color: #666;
            font-style: italic;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .loading:after {
            content: "...";
            animation: dots 1.5s steps(5, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: "."; }
            40% { content: ".."; }
            60%, 100% { content: "..."; }
        }
        .error {
            color: #d32f2f;
            font-weight: bold;
            padding: 8px;
            border-radius: 4px;
            background-color: #ffebee;
            border: 1px solid #ffcdd2;
        }
        select {
            padding: 10px;
            font-size: 16px;
            margin: 10px 0;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            background-color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        select:hover {
            border-color: #2196F3;
        }
        .mode-selection, .model-selection {
            margin: 15px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .mode-info, .model-info {
            margin-top: 8px;
            font-size: 14px;
            color: #666;
            line-height: 1.4;
        }
        .section-title {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        .mode-selection {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .mode-info {
            margin-top: 5px;
            font-size: 14px;
            color: #666;
        }
        #editMode {
            padding: 8px;
            font-size: 16px;
            border-radius: 5px;
            border: 2px solid #ddd;
            width: 100%;
            max-width: 300px;
        }
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-top: 20px;
        }
        .result-box {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
        }
        .result-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .edited-text {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 0;
            font-size: 16px;
            line-height: 1.6;
        }
        .explanation {
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 15px;
            margin: 0;
            font-size: 14px;
            line-height: 1.6;
            color: #1976D2;
        }
    </style>
</head>
<body>
    <h1>Persian Text Editor and Translator</h1>
    <div class="container">
        <!-- Persian Text Editor Section -->
        <div class="text-section">
            <div class="section-title">Persian Text Editor</div>
            <div class="mode-selection">
                <select id="editMode">
                    <option value="fast">Fast Edit (Grammar & Spelling)</option>
                    <option value="detailed">Detailed Edit (Professional & Coaching)</option>
                </select>
                <div class="mode-info">Fast mode: Quick grammar and spelling fixes. Detailed mode: Deep professional content enhancement.</div>
            </div>
            <div class="model-selection">
                <select id="editModel">
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Good)</option>
                    <option value="gpt-4">GPT-4 (Most Accurate)</option>
                    <option value="claude-3">Claude-3 (Accurate)</option>
                    <option value="gemini">Gemini Pro (Advanced AI)</option>
                </select>
                <div id="editModelInfo" class="model-info"></div>
            </div>
            <textarea id="persianText" placeholder="Write your Persian text here..." dir="rtl"></textarea>
            <button id="editButton" class="edit">Edit Text</button>
            
            <div class="results-container">
                <div class="result-box">
                    <div class="result-title">Edited Text:</div>
                    <div id="editedText" class="edited-text" dir="rtl"></div>
                </div>
                <div class="result-box">
                    <div class="result-title">Explanation:</div>
                    <div id="explanation" class="explanation"></div>
                </div>
            </div>
        </div>

        <!-- Translation Section -->
        <div class="text-section">
            <div class="section-title">Translate to English</div>
            <div class="model-selection">
                <select id="model">
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Good)</option>
                    <option value="gpt-4">GPT-4 (Most Accurate)</option>
                    <option value="claude-3">Claude-3 (Accurate)</option>
                    <option value="google-translate">Google Cloud Translation (Basic)</option>
                    <option value="gemini">Gemini Pro (Advanced AI)</option>
                </select>
                <div id="modelInfo" class="model-info"></div>
            </div>
            <button id="translateButton">Translate</button>
            <div id="result"></div>
        </div>
    </div>

    <script>
        // Wait for DOM to load
        window.addEventListener("DOMContentLoaded", () => {
            // Initialize model info
            updateModelInfo();

            // Add event listeners
            document.getElementById("editButton").addEventListener("click", editText);
            document.getElementById("translateButton").addEventListener("click", translateText);
            document.getElementById("model").addEventListener("change", updateModelInfo);
            document.getElementById("editModel").addEventListener("change", updateModelInfo);
            document.getElementById("persianText").addEventListener("keypress", (e) => {
                if (e.key === "Enter" && e.ctrlKey) {
                    e.preventDefault();
                    editText();
                }
            });
        });

        // Update model information display
        function updateModelInfo() {
            const model = document.getElementById("model").value;
            const modelInfo = document.getElementById("modelInfo");
            const editModel = document.getElementById("editModel").value;
            const editModelInfo = document.getElementById("editModelInfo");
            
            const modelDescriptions = {
                "gpt-3.5-turbo": "Fast and reliable processing with good accuracy",
                "gpt-4": "Most accurate processing, better understanding of context and nuances",
                "claude-3": "Advanced AI model with strong understanding of Persian language",
                "google-translate": "Basic machine translation, good for simple texts",
                "gemini": "Google's advanced AI model, excellent for context and cultural nuances"
            };
            
            modelInfo.textContent = modelDescriptions[model];
            editModelInfo.textContent = modelDescriptions[editModel];
        }

        // Handle text editing
        async function editText() {
            const text = document.getElementById("persianText").value;
            const editMode = document.getElementById("editMode").value;
            const editModel = document.getElementById("editModel").value;
            const editedText = document.getElementById("editedText");
            const explanation = document.getElementById("explanation");
            
            if (!text.trim()) {
                editedText.innerHTML = '<span class="error">لطفا متن فارسی را وارد کنید</span>';
                explanation.innerHTML = "";
                return;
            }
            
            editedText.innerHTML = '<span class="loading">در حال ویرایش متن...</span>';
            explanation.innerHTML = "";
            
            try {
                const response = await fetch("/edit", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ 
                        text: text,
                        mode: editMode,
                        model: editModel
                    })
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || "Edit failed");
                }
                
                const data = await response.json();
                editedText.textContent = data.edited_text;
                explanation.textContent = data.explanation;
            } catch (error) {
                editedText.innerHTML = `<span class="error">خطا: ${error.message}</span>`;
                explanation.innerHTML = "";
                console.error("Edit error:", error);
            }
        }

        // Handle text translation
        async function translateText() {
            const text = document.getElementById("editedText").textContent || document.getElementById("persianText").value;
            const model = document.getElementById("model").value;
            const result = document.getElementById("result");
            
            if (!text.trim()) {
                result.innerHTML = '<span class="error">Please enter some text to translate</span>';
                return;
            }
            
            result.innerHTML = '<span class="loading">Translating...</span>';
            
            try {
                const response = await fetch("/translate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        text: text,
                        model: model
                    })
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || "Translation failed");
                }
                
                const data = await response.json();
                result.innerHTML = data.translated_text;
            } catch (error) {
                result.innerHTML = `<span class="error">❌ Error: ${error.message}</span>`;
                console.error("Translation error:", error);
            }
        }
    </script>
</body>
</html>
''')

class EditRequest(BaseModel):
    text: str
    mode: str  # "fast" or "detailed"
    model: ModelType = ModelType.GPT35  # Default to GPT-3.5-turbo if not specified

class TranslationRequest(BaseModel):
    text: str
    model: ModelType

def get_system_prompt(mode: str, language: str) -> str:
    return f"""You are an expert bilingual editor specializing in {language} professional development, coaching, and psychological content.

EXPERTISE AREAS:
1. Professional Terminology:
   - Coaching and psychological terms
   - Leadership development vocabulary
   - Self-help and motivational language
   - Technical accuracy in both languages

2. Content Style:
   - Professional yet engaging tone
   - Clear and authoritative voice
   - Appropriate formality level
   - Cultural sensitivity

3. Genre-Specific Knowledge:
   - Self-help book standards
   - Coaching methodology
   - Psychological concepts
   - Motivational writing

MODE SPECIFICATIONS:
For '{mode}' mode in {language}:
- Fast Mode: 
  • Basic grammar and spelling fixes
  • Simple terminology corrections
  • Fix punctuation and spacing
  • Quick surface improvements
  • Minimal technical adjustments

- Detailed Mode:
  • Precise professional terminology refinement
  • Careful tone adjustment for coaching context
  • Enhanced clarity without restructuring
  • Professional language improvement
  • Coaching terminology accuracy
  • Cultural nuance preservation
  • Technical term precision
  • Maintain original message and structure
  • Polish existing expressions
  • Subtle improvements to flow

LANGUAGE-SPECIFIC GUIDELINES:
For Persian (فارسی):
- حفظ لحن رسمی و حرفه‌ای
- استفاده صحیح از اصطلاحات تخصصی کوچینگ
- رعایت ساختار نگارشی متون روانشناسی
- حفظ انسجام در ترجمه مفاهیم تخصصی

For English:
- Maintain professional coaching terminology
- Ensure psychological concept accuracy
- Preserve motivational impact
- Balance technical and accessible language"""

def get_edit_prompt(text: str, mode: str, language: str) -> str:
    return f"""Edit this {language} professional development text in {mode} mode.

EDITING GUIDELINES:
1. Maintain subject matter expertise in:
   - Coaching methodology
   - Psychological concepts
   - Professional development
   - Leadership principles

2. Ensure appropriate:
   - Technical terminology
   - Professional tone
   - Concept clarity
   - Engagement level

3. Preserve:
   - Core message integrity
   - Professional credibility
   - Motivational impact
   - Cultural context

Text to edit:
{text}

Return only the edited text without explanations."""

def get_explanation_prompt(original_text: str, edited_text: str, language: str) -> str:
    return f"""Provide a brief, bullet-point summary of key changes made to this {language} text:

• Grammar & Style:
  - List 1-2 major grammar/style improvements

• Terminology:
  - Note any professional term improvements

• Tone & Impact:
  - Mention key tone/impact enhancements

Keep the explanation short and focused on the most important changes.

Original Text:
{original_text}

Edited Text:
{edited_text}

Provide a concise bullet-point summary."""

async def translate_with_openai(text: str, model: str) -> str:
    try:
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key is missing")
            
        temperature = 0.1  # Very low temperature for exact translations

        system_prompt = """You are an expert translator specializing in professional development, coaching, and psychological content. 

IMPORTANT: This is a direct translation task. Focus on:
- Exact meaning preservation
- Precise terminology mapping
- Professional coaching and psychological terms
- No creative variations or generations
- Maintain source text structure when possible"""

        user_prompt = f"""Translate this text while maintaining exact meaning and terminology:
1. Use precise professional/coaching terms
2. Maintain source text structure
3. Preserve exact concepts
4. Keep cultural context
5. No creative additions

Text to translate:
{text}

Provide only the direct translation."""

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {str(e)}")
        raise Exception(f"OpenAI translation failed: {str(e)}")

async def translate_with_claude(text: str) -> str:
    try:
        if not ANTHROPIC_API_KEY:
            raise Exception("Anthropic API key is missing")
            
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": ANTHROPIC_API_KEY
        }
        
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "temperature": 0.1,  # Very low for exact translations
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise translator focusing on exact terminology and accurate translations. Maintain maximum accuracy and conciseness."
                },
                {
                    "role": "user",
                    "content": f"Translate this Persian text to English with precise terminology and maximum accuracy: {text}"
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["content"][0]["text"].strip()
        
    except Exception as e:
        logger.error(f"Claude translation error: {str(e)}")
        raise Exception(f"Claude translation failed: {str(e)}")

async def translate_with_google_cloud(text: str) -> str:  # Renamed from translate_with_gemini
    try:
        logger.info("Starting Google Cloud Translation")
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
        logger.info(f"Google Cloud Translation successful: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Google Cloud Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google Cloud Translation failed: {str(e)}")

async def translate_with_gemini(text: str) -> str:
    try:
        if not GEMINI_API_KEY:
            raise Exception("Gemini API key is missing")
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        generation_config = {
            "temperature": 0.1  # Low temperature for accurate translation
        }
        
        prompt = f"""You are an expert translator specializing in professional development and coaching content. 

Task: Translate the following Persian text to English while maintaining:
1. Professional coaching terminology
2. Psychological concept accuracy
3. Motivational impact
4. Cultural context
5. Professional tone

Focus on EXACT translation with precise terminology matching. This is a translation task, not text generation.

Text to translate:
{text}

Provide only the direct translation, maintaining exact meaning."""
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not hasattr(response, 'text'):
            raise Exception("No response from Gemini")
            
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini translation error: {str(e)}")
        raise Exception(f"Gemini translation failed: {str(e)}")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        logger.info(f"Attempting to translate text using {request.model}: {request.text}")
        
        if request.model == ModelType.GPT35:
            translated_text = await translate_with_openai(request.text, "gpt-3.5-turbo")
        elif request.model == ModelType.GPT4:
            translated_text = await translate_with_openai(request.text, "gpt-4")
        elif request.model == ModelType.CLAUDE:
            translated_text = await translate_with_claude(request.text)
        elif request.model == ModelType.GOOGLE:
            translated_text = await translate_with_google_cloud(request.text)
        elif request.model == ModelType.GEMINI:
            translated_text = await translate_with_gemini(request.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified")
            
        logger.info(f"Translation successful with {request.model}")
        return {"translated_text": translated_text}
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit")
async def edit(request: EditRequest):
    try:
        logger.info(f"Attempting to edit text in {request.mode} mode using {request.model}: {request.text}")
        
        # Detect language
        language = "Persian" if any("\u0600" <= c <= "\u06FF" for c in request.text) else "English"
        
        system_prompt = get_system_prompt(request.mode, language)
        user_prompt_edit = get_edit_prompt(request.text, request.mode, language)
        user_prompt_explain = get_explanation_prompt(request.text, request.mode, language)
        
        if request.model in [ModelType.GPT35, ModelType.GPT4]:
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key is missing")
                
            # Set temperature based on model and mode
            if request.model == ModelType.GPT4:
                # GPT-4 temperatures
                if request.mode == "detailed":
                    temperature = 0.4  # Higher for detailed editing to allow creative improvements
                else:
                    temperature = 0.2  # Lower for fast mode focusing on accuracy
            else:
                # GPT-3.5 temperatures
                if request.mode == "detailed":
                    temperature = 0.5  # Higher for detailed editing
                else:
                    temperature = 0.3  # Lower for fast mode

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # First get the edited text
            data = {
                "model": request.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_edit}
                ],
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            edited_text = result["choices"][0]["message"]["content"].strip()
            
            # Then get the explanation
            data["messages"][1]["content"] = user_prompt_explain + edited_text
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["choices"][0]["message"]["content"].strip()
            
        elif request.model == ModelType.CLAUDE:
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key is missing")
                
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": ANTHROPIC_API_KEY
            }
            
            # First get the edited text
            data = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 1024,
                "temperature": 0.4 if request.mode == "detailed" else 0.2,  # Adjust based on mode
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt_edit
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            edited_text = result["content"][0]["text"].strip()
            
            # Then get the explanation
            data["messages"][1]["content"] = user_prompt_explain + edited_text
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["content"][0]["text"].strip()
            
        elif request.model == ModelType.GEMINI:
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini API key is missing")
                
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            generation_config = {
                "temperature": 0.25 if request.mode == "detailed" else 0.2  # Very slight increase for detailed mode to allow minor refinements
            }
            
            # First get the edited text
            prompt = f"{system_prompt}\n\n{user_prompt_edit}"
            response = model.generate_content(prompt)
            
            if not hasattr(response, 'text'):
                raise Exception("No response from Gemini")
            edited_text = response.text.strip()
            
            # Then get the explanation
            prompt = f"{system_prompt}\n\n{user_prompt_explain}{edited_text}"
            response = model.generate_content(prompt)
            
            if not hasattr(response, 'text'):
                raise Exception("No explanation from Gemini")
            explanation = response.text.strip()
            
        elif request.model == ModelType.GOOGLE:
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key is missing")
                
            url = "https://translation.googleapis.com/language/translate/v2"
            # First translate to English
            params = {
                'q': request.text,
                'target': 'en',
                'source': 'fa',
                'key': GOOGLE_API_KEY
            }
            response = requests.post(url, params=params)
            response.raise_for_status()
            english = response.json()['data']['translations'][0]['translatedText']
            
            # Then back to Persian
            params['q'] = english
            params['target'] = 'fa'
            params['source'] = 'en'
            response = requests.post(url, params=params)
            response.raise_for_status()
            edited_text = response.json()['data']['translations'][0]['translatedText']
            
            # Generate a simple explanation
            explanation = "Text was processed through translation to English and back to Persian for basic improvements."
        
        logger.info(f"Edit successful with {request.model}: {edited_text}")
        logger.info(f"Explanation: {explanation}")
        return {"edited_text": edited_text, "explanation": explanation}
            
    except Exception as e:
        logger.error(f"Edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def test_gemini_connection() -> bool:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        return hasattr(response, 'text')
    except Exception as e:
        logger.error(f"Error testing Gemini connection: {str(e)}")
        return False

# Add a route to test the connection
@app.get("/test-gemini")
async def test_gemini():
    """Endpoint to test Gemini API connection"""
    success = await test_gemini_connection()
    if success:
        return {"status": "success", "message": "Gemini API connection test passed"}
    else:
        raise HTTPException(status_code=500, detail="Gemini API connection test failed")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Test Gemini connection before starting the server
    asyncio.run(test_gemini_connection())
    
    uvicorn.run(app, host="0.0.0.0", port=8088)