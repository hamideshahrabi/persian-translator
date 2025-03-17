from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Tuple
import os
from dotenv import load_dotenv
import logging
import json
from enum import Enum
import google.generativeai as genai
import difflib
import re
import openai
import asyncio
from functools import lru_cache
from difflib import SequenceMatcher
from html import escape
from contextlib import asynccontextmanager
import socket

# Configure logging and load API keys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate API keys
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found")
    raise ValueError("OpenAI API key is required")

if not GEMINI_API_KEY:
    logger.error("Gemini API key not found")
    raise ValueError("Gemini API key is required")

# Updated prompt for more aggressive editing
EDIT_PROMPT_PERSIAN = """لطفاً متن فارسی زیر را با دقت بالا و با رعایت تمام نکات زیر ویرایش نمایید:

اصول اساسی ویرایش:
۱. بررسی دقیق گرامر و نحو:
   - اطمینان از صحت ساختار تمام جملات
   - بررسی تطابق فعل و فاعل
   - استفاده صحیح از حروف ربط و اضافه
   - رعایت زمان افعال و هماهنگی آنها

۲. ارتقای سطح واژگان:
   - جایگزینی واژگان عامیانه با معادل‌های رسمی و تخصصی
   - استفاده از واژگان دقیق و تخصصی حوزه مربیگری
   - حذف کلمات زائد و تکراری
   - انتخاب واژگان متناسب با سطح آکادمیک

۳. ساختار متن:
   - هر پاراگراف باید شامل جملات کامل باشد
   - تنها عناوین می‌توانند تک‌کلمه یا عبارت باشند
   - تمام توضیحات باید در قالب جملات کامل ارائه شوند
   - رعایت پیوستگی و انسجام بین جملات

۴. سطح رسمی و حرفه‌ای:
   - حفظ لحن کاملاً رسمی و آکادمیک
   - استفاده از ساختارهای نگارشی استاندارد
   - رعایت اصول نگارش علمی
   - حذف هرگونه عبارت غیررسمی یا محاوره‌ای

۵. دقت در محتوا:
   - حفظ دقیق معنا و مفهوم اصلی متن
   - تقویت وضوح و شفافیت پیام
   - اطمینان از انتقال صحیح مفاهیم تخصصی
   - حفظ تمام نکات کلیدی متن اصلی

قوانین سخت‌گیرانه:
- هر بخش متن (به جز عناوین) باید شامل جملات کامل باشد
- هیچ کلمه یا عبارت منفردی نباید بدون ساختار جمله باقی بماند
- تمام جملات باید از نظر گرامری کاملاً صحیح باشند
- هر تغییر در واژگان باید منجر به ارتقای سطح متن شود
- حفظ معنای دقیق متن اصلی الزامی است

لطفاً متن را با رعایت تمام این نکات ویرایش نمایید."""

# Initialize API clients with optimized settings and better error handling
async def check_api_connections():
    """Check connections to OpenAI and Gemini APIs."""
    openai_status = "❌ Not Connected"
    gemini_status = "❌ Not Connected"
    
    try:
        # Test OpenAI connection
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        if response:
            openai_status = "✅ Connected"
    except Exception as e:
        openai_status = f"❌ Error: {str(e)}"
    
    try:
        # Test Gemini connection with proper model
        response = gemini_model.generate_content("Test connection")
        if response and response.text:
            gemini_status = "✅ Connected"
        else:
            gemini_status = "❌ Error: Empty response"
    except Exception as e:
        gemini_status = f"❌ Error: {str(e)}"
    
    return openai_status, gemini_status

try:
    openai_client = openai.AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=30.0,
        max_retries=3
    )

    genai.configure(api_key=GEMINI_API_KEY)
    
    # List available models
    try:
        available_models = genai.list_models()
        logger.info("Available Gemini models:")
        for model in available_models:
            logger.info(f"- {model.name}")
    except Exception as e:
        logger.error(f"Failed to list Gemini models: {str(e)}")
    
    # Initialize Gemini model with proper configuration
    gemini_model = genai.GenerativeModel(
        model_name='models/gemini-1.5-pro',
        generation_config={
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
    )
    
    # Test Gemini model
    try:
        test_response = gemini_model.generate_content("Test connection.")
        if test_response and test_response.text:
            logger.info("✅ Successfully tested Gemini model connection")
        else:
            logger.error("❌ Gemini model returned empty response")
    except Exception as e:
        logger.error(f"❌ Failed to test Gemini model: {str(e)}")
        raise

except Exception as e:
    logger.error(f"Failed to initialize API clients: {str(e)}")
    raise

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4-turbo-preview"
    GEMINI = "models/gemini-1.5-pro"

class EditRequest(BaseModel):
    text: str
    model: str

class Change(BaseModel):
    type: str
    old: Optional[str] = None
    new: Optional[str] = None

class EditResponse(BaseModel):
    edited_text: str
    technical_explanation: str
    changes: List[Change]
    diff_html: str

class StatusResponse(BaseModel):
    openai_status: str
    gemini_status: str
    server_status: str = "✅ Running"

def check_content_preserved(original_text: str, edited_text: str, threshold: float = 0.3) -> bool:
    """Check if the edited text preserves core meaning while allowing substantial reformulation."""
    def clean_text(text: str) -> str:
        # Basic cleaning while preserving word boundaries
        text = re.sub(r'[\u200c\u200f]', '', text)  # Remove invisible characters
        text = re.sub(r'[^\u0600-\u06FF\s]+', ' ', text)  # Keep Persian characters and spaces
        return ' '.join(text.split())  # Normalize spaces
    
    original_cleaned = clean_text(original_text)
    edited_cleaned = clean_text(edited_text)
    
    # Split into sentences for comparison
    def get_sentences(text):
        return [s.strip() for s in re.split(r'[.!?؟।۔।؛]', text) if s.strip()]
    
    original_sentences = get_sentences(original_cleaned)
    edited_sentences = get_sentences(edited_cleaned)
    
    # Check if all key concepts from original are present in edited
    original_words = set(w for s in original_sentences for w in s.split())
    edited_words = set(w for s in edited_sentences for w in s.split())
    
    # Calculate word preservation ratio with lower threshold
    preserved_words = len(original_words.intersection(edited_words))
    word_preservation_ratio = preserved_words / len(original_words) if original_words else 1.0
    
    # More lenient sentence count ratio
    sentence_ratio = len(edited_sentences) / len(original_sentences) if original_sentences else 1.0
    
    # Lowered thresholds for better flexibility
    return word_preservation_ratio >= 0.2 and 0.5 <= sentence_ratio <= 1.5

async def process_gemini_edit(text: str) -> str:
    """Process text editing using Gemini API with strict grammar and sentence completion."""
    if not text.strip():
        raise ValueError("Empty text provided")

    if len(text) > 60000:
        raise ValueError("Text too long for Gemini API (max 60000 characters)")

    # Pre-process text to identify incomplete sentences and words
    def format_text_for_editing(text: str) -> str:
        sentences = [s.strip() for s in re.split(r'[.!?؟।۔।؛]', text) if s.strip()]
        formatted_text = ""
        for sentence in sentences:
            if any(word.endswith('‌') for word in sentence.split()):
                formatted_text += f"[نیاز به تکمیل کلمات]: {sentence}\n"
            elif not any(sentence.endswith(p) for p in ['.', '!', '?', '؟', '۔']):
                formatted_text += f"[نیاز به تکمیل جمله]: {sentence}\n"
            else:
                formatted_text += f"{sentence}.\n"
        return formatted_text.strip()

    formatted_text = format_text_for_editing(text)
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            prompt = f"""لطفاً متن زیر را با دقت ویرایش کنید. به موارد زیر توجه ویژه نمایید:

۱. تکمیل کلمات و جملات:
   - عبارات با علامت [نیاز به تکمیل کلمات] دارای کلمات ناقص هستند که باید تکمیل شوند
   - عبارات با علامت [نیاز به تکمیل جمله] باید به جملات کامل تبدیل شوند
   - هر جمله باید معنای کامل و مستقل داشته باشد

۲. حفظ محتوا:
   - هیچ جمله‌ای نباید حذف شود مگر با جایگزین مناسب
   - تمام مفاهیم اصلی باید در متن نهایی وجود داشته باشند
   - هر جمله اصلی باید حداقل یک معادل در متن ویرایش شده داشته باشد

۳. بهبود کیفیت:
   - استفاده از واژگان تخصصی و رسمی
   - اصلاح ساختار جملات
   - حفظ انسجام متن

متن اصلی برای ویرایش:
{formatted_text}

لطفاً متن را طوری ویرایش کنید که:
۱. تمام کلمات ناقص تکمیل شوند
۲. هر جمله کامل و معنادار باشد
۳. هیچ محتوایی بدون جایگزین حذف نشود
۴. سطح نگارش و واژگان ارتقا یابد"""

            logger.info(f"Attempting Gemini API call (attempt {attempt + 1}/{max_retries})")
            
            edit_model = genai.GenerativeModel(
                model_name='models/gemini-1.5-pro',
                generation_config={
                    'temperature': 0.5,
                    'top_p': 0.9,
                    'top_k': 30,
                    'max_output_tokens': 8192,
                }
            )
            
            response = await asyncio.wait_for(
                asyncio.to_thread(lambda: edit_model.generate_content(prompt)),
                timeout=30.0
            )
            
            if not response or not response.text:
                last_error = ValueError("Empty response from Gemini")
                if attempt == max_retries - 1:
                    raise last_error
                await asyncio.sleep(1)
                continue
                
            edited_text = response.text.strip()
            
            # Verify no incomplete words
            edited_sentences = [s.strip() for s in re.split(r'[.!?؟।۔।؛]', edited_text) if s.strip()]
            for sentence in edited_sentences:
                if any(word.endswith('‌') for word in sentence.split()):
                    last_error = ValueError("Some words are still incomplete")
                    if attempt == max_retries - 1:
                        raise last_error
                    await asyncio.sleep(1)
                    continue
            
            # Check content preservation with lower threshold
            if not check_content_preserved(text, edited_text, threshold=0.2):
                last_error = ValueError("Failed to preserve content")
                if attempt == max_retries - 1:
                    raise last_error
                await asyncio.sleep(1)
                continue
            
            logger.info("Successfully processed text with Gemini")
            return edited_text

        except asyncio.TimeoutError:
            last_error = TimeoutError("Gemini API timeout")
            if attempt == max_retries - 1:
                raise last_error
            await asyncio.sleep(2)
        except Exception as e:
            last_error = e
            logger.error(f"Gemini API error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2)
    
    raise last_error

async def process_openai_edit(text: str, model: str = "gpt-3.5-turbo") -> str:
    """Process text editing using OpenAI API with strict grammar and sentence completion."""
    if not text.strip():
        raise ValueError("Empty text provided")
            
    if len(text) > 50000:
        raise ValueError("Text too long for OpenAI API (max 50000 characters)")

    # Pre-process text to identify incomplete sentences and quotes
    def format_text_for_editing(text: str) -> str:
        # First, identify and mark quotes
        quote_pattern = r'"[^"]+"|«[^»]+»|"[^"]+"'
        quotes = re.findall(quote_pattern, text)
        text_with_marked_quotes = text
        for i, quote in enumerate(quotes):
            text_with_marked_quotes = text_with_marked_quotes.replace(quote, f"[نقل قول {i+1}]: {quote}")

        sentences = [s.strip() for s in re.split(r'[.!?؟।۔।؛]', text_with_marked_quotes) if s.strip()]
        formatted_text = ""
        for sentence in sentences:
            if "[نقل قول" in sentence:
                formatted_text += f"{sentence}.\n"
            elif any(word.endswith('‌') for word in sentence.split()) or not any(sentence.endswith(p) for p in ['.', '!', '?', '؟', '۔']):
                formatted_text += f"[نیازمند تکمیل و اصلاح]: {sentence}\n"
            else:
                formatted_text += f"{sentence}.\n"
        return formatted_text.strip()

    formatted_text = format_text_for_editing(text)

    # Adjust parameters based on model
    if model == "gpt-4-turbo-preview":
        temperature = 0.4  # Increased temperature for more flexibility
        presence_penalty = 0.1
        frequency_penalty = 0.1
        top_p = 0.9
        threshold = 0.2  # Lowered threshold for GPT-4
    else:  # gpt-3.5-turbo
        temperature = 0.5  # Increased temperature for more flexibility
        presence_penalty = 0.2
        frequency_penalty = 0.2
        top_p = 0.9
        threshold = 0.2  # Lowered threshold for GPT-3.5

    # Enhanced system prompt for better sentence handling
    system_prompt = """شما یک ویراستار متخصص و سخت‌گیر هستید که با دقت بالا متون را ویرایش می‌کند. وظایف اصلی شما:

۱. تکمیل و اصلاح جملات:
   - هر عبارت با علامت [نیازمند تکمیل و اصلاح] باید به جمله کامل تبدیل شود
   - کلمات ناقص باید تکمیل شوند
   - هر جمله باید معنای کامل داشته باشد
   - هیچ جمله‌ای نباید حذف شود، مگر آنکه با جمله بهتری جایگزین شود

۲. حفظ و بهبود محتوا:
   - تمام مفاهیم اصلی متن باید حفظ شوند
   - هر جمله باید حداقل یک معادل در متن ویرایش شده داشته باشد
   - بهبود جملات بدون حذف محتوای اصلی
   - اگر جمله‌ای نیاز به تغییر دارد، باید با جمله بهتری جایگزین شود

۳. حفظ نقل قول‌ها:
   - عبارات با علامت [نقل قول] باید دقیقاً حفظ شوند
   - محتوای داخل نقل قول‌ها نباید تغییر کند
   - فقط ساختار جملات اطراف نقل قول‌ها می‌تواند بهبود یابد

۴. ارتقای کیفیت:
   - اصلاح گرامر و نگارش
   - استفاده از واژگان تخصصی و رسمی
   - حفظ انسجام و پیوستگی متن
   - بهبود ساختار جملات

قوانین مهم:
- هیچ جمله‌ای نباید بدون جایگزین حذف شود
- هر کلمه ناقص باید تکمیل شود
- هر جمله باید کامل و معنادار باشد
- محتوای اصلی و نقل قول‌ها باید دقیقاً حفظ شوند
- هر پاراگراف باید با جمله کامل پایان یابد"""

    # Add specific instructions for handling the text
    user_prompt = f"""لطفاً متن زیر را ویرایش کنید. به نکات زیر توجه ویژه نمایید:

۱. جملات ناقص و نیازمند تکمیل را حتماً به صورت کامل بازنویسی کنید
۲. نقل قول‌ها را دقیقاً حفظ کنید
۳. هیچ جمله‌ای را بدون جایگزین مناسب حذف نکنید
۴. اطمینان حاصل کنید که هر پاراگراف با جمله کامل پایان می‌یابد

متن برای ویرایش:
{formatted_text}"""

    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=len(text.split()) * 2,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_p=top_p
    )
    
    if not response.choices:
        raise ValueError("No response from OpenAI API")
        
    edited_text = response.choices[0].message.content.strip()
    
    if not edited_text:
        raise ValueError("Empty response from OpenAI API")
    
    # Verify all sentences are complete and quotes are preserved
    def verify_text(edited_text: str, original_text: str) -> bool:
        # Check for incomplete sentences
        sentences = [s.strip() for s in re.split(r'[.!?؟।۔।؛]', edited_text) if s.strip()]
        for sentence in sentences:
            if any(word.endswith('‌') for word in sentence.split()):
                return False
            
        # Check that all quotes from original text are present in edited text
        original_quotes = set(re.findall(r'"[^"]+"|«[^»]+»|"[^"]+"', original_text))
        edited_quotes = set(re.findall(r'"[^"]+"|«[^»]+»|"[^"]+"', edited_text))
        
        if not original_quotes.issubset(edited_quotes):
            return False
            
        return True
    
    if not verify_text(edited_text, text):
        raise ValueError("Text verification failed: incomplete sentences or missing quotes")
    
    # Check content preservation with new threshold
    if not check_content_preserved(text, edited_text, threshold=threshold):
        raise ValueError("Failed to preserve content while editing")
        
    return edited_text

@lru_cache(maxsize=1000)
def detect_changes(original_text: str, edited_text: str) -> List[Change]:
    """Detect changes between original and edited text with caching."""
    changes = []
    
    if original_text == edited_text:
        return changes
    
    matcher = difflib.SequenceMatcher(None, original_text, edited_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            changes.append(Change(type="grammar", old=original_text[i1:i2], new=edited_text[j1:j2]))
        elif tag == 'delete':
            changes.append(Change(type="removed", old=original_text[i1:i2]))
        elif tag == 'insert':
            changes.append(Change(type="added", new=edited_text[j1:j2]))
    
    return changes

def generate_diff_html(text: str, edited_text: str) -> str:
    """Generate HTML diff between original and edited text."""
    diff_html = []
    matcher = SequenceMatcher(None, text, edited_text)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            diff_html.append(text[i1:i2])
        elif tag in ('delete', 'replace'):
            diff_html.append(f'<span class="diff-removed-text">{text[i1:i2]}</span>')
        if tag in ('insert', 'replace'):
            diff_html.append(f'<span class="diff-added-text">{edited_text[j1:j2]}</span>')
    
    return ''.join(diff_html)

# Setup API
app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root(request: Request):
    # Get current API status
    status = await check_status()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_status": status
    })

@app.post("/edit")
async def edit(request: EditRequest) -> EditResponse:
    """Edit Persian text using the specified model."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
            
        edited_text = text  # Initialize with original text
        explanation = ""
        error_occurred = False
        
        try:
            logger.info(f"Received edit request with model: {request.model}")
            
            if request.model == ModelType.GEMINI.value:
                logger.info(f"Attempting to edit with Gemini model")
                edited_text = await process_gemini_edit(text)
                explanation = "✅ Used Gemini for editing"
            elif request.model == ModelType.GPT35.value:
                logger.info(f"Attempting to edit with GPT-3.5 model")
                edited_text = await process_openai_edit(text, model=ModelType.GPT35.value)
                explanation = "✅ Used GPT-3.5 for editing"
            elif request.model == ModelType.GPT4.value:
                logger.info(f"Attempting to edit with GPT-4 model")
                edited_text = await process_openai_edit(text, model=ModelType.GPT4.value)
                explanation = "✅ Used GPT-4 for editing"
            else:
                logger.error(f"Unsupported model requested: {request.model}")
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
        except Exception as e:
            error_occurred = True
            error_msg = str(e)
            logger.error(f"Edit error with {request.model}: {error_msg}")
            
            if "404" in error_msg:
                raise HTTPException(status_code=400, detail=f"Model {request.model} is currently unavailable. Please try a different model.")
            elif "timeout" in error_msg.lower():
                raise HTTPException(status_code=408, detail="The request timed out. Please try again or use a different model.")
            elif "api key" in error_msg.lower():
                raise HTTPException(status_code=401, detail="Invalid API key. Please contact support.")
            else:
                raise HTTPException(status_code=400, detail=str(e))
            
        if edited_text == text and not error_occurred:
            explanation = f"⚠️ No changes were made to the text. The model either found no improvements needed or failed to make meaningful changes while preserving the content."
        elif edited_text != text:
            explanation = f"✅ Successfully applied text improvements"
            
            # Generate changes and diff HTML
            changes = detect_changes(text, edited_text)
            diff_html = generate_diff_html(text, edited_text)
            
            logger.info(f"Successfully processed edit request. Changes detected: {len(changes)}")
            
            return EditResponse(
                edited_text=edited_text,
                technical_explanation=explanation,
                changes=changes,
                diff_html=diff_html
            )
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Edit error: {error_msg}")
        user_friendly_msg = "An unexpected error occurred"
        
        if isinstance(e, HTTPException):
            user_friendly_msg = e.detail
        elif "api key" in error_msg.lower():
            user_friendly_msg = "API key error - Please contact support"
        elif "timeout" in error_msg.lower():
            user_friendly_msg = "Request timed out - Please try again"
        elif "too many requests" in error_msg.lower():
            user_friendly_msg = "Too many requests - Please wait a moment and try again"
        
        return EditResponse(
            edited_text=text,
            technical_explanation=f"❌ {user_friendly_msg}",
            changes=[],
            diff_html=text
        )

@app.get("/status")
async def check_status() -> StatusResponse:
    """Check the status of all API connections."""
    openai_status, gemini_status = await check_api_connections()
    return StatusResponse(
        openai_status=openai_status,
        gemini_status=gemini_status
    )

@app.post("/translate")
async def translate_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model_type = data.get("model", "GPT35")
        
        if model_type == "GEMINI":
            # Use Gemini for translation with proper prompt
            model = genai.GenerativeModel("models/gemini-1.5-pro")
            prompt = f"Please translate the following text to English. Keep the translation accurate and natural:\n\n{text}"
            response = model.generate_content(prompt)
            return {"translated_text": response.text}
        else:
            # Use OpenAI for translation with proper prompt
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate the given text to English accurately and naturally."},
                {"role": "user", "content": text}
            ]
            
            model_name = "gpt-3.5-turbo" if model_type == "GPT35" else "gpt-4-turbo-preview"
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return {"translated_text": response.choices[0].message.content}
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import socket
    
    PORT = 8088
    MAX_PORT_ATTEMPTS = 10
    
    # Try to find an available port
    for port_attempt in range(PORT, PORT + MAX_PORT_ATTEMPTS):
        try:
            # Test if port is in use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port_attempt))
                PORT = port_attempt
                break
        except OSError:
            if port_attempt == PORT + MAX_PORT_ATTEMPTS - 1:
                logger.error(f"Could not find an available port in range {PORT}-{PORT + MAX_PORT_ATTEMPTS}")
                raise
            continue
    
    logger.info(f"Starting server on port {PORT}")
    
    # Check API connections
    openai_status, gemini_status = asyncio.run(check_api_connections())
    logger.info("=== API Connection Status ===")
    logger.info(f"OpenAI API: {openai_status}")
    logger.info(f"Gemini API: {gemini_status}")
    logger.info("==========================")
    
    uvicorn.run(
        "translation_bot:app",
        host="0.0.0.0",
        port=PORT,
        reload=True
    )