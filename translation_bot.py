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

# Initialize global API clients
openai_client = None
try:
    openai_client = openai.AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=30.0,
        max_retries=3
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

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

# Initialize API clients and test connections
async def test_api_connections():
    openai_status = "Not tested"
    gemini_status = "Not tested"
    
    try:
        # Test OpenAI connection
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        if response:
            logger.info("✅ Successfully tested OpenAI API connection")
            openai_status = "✅ Connected"
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        openai_status = f"❌ Error: {str(e)}"
    
    try:
        # Initialize Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models
        logger.info("Available Gemini models:")
        for m in genai.list_models():
            logger.info(f"- {m.name}")
        
        # Test Gemini connection
        model = genai.GenerativeModel('models/gemini-1.5-pro')
        response = model.generate_content("Test")
        if response:
            logger.info("✅ Successfully tested Gemini model connection")
            gemini_status = "✅ Connected"
        else:
            logger.error("❌ Gemini model returned empty response")
            gemini_status = "❌ Error: Empty response"
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {str(e)}")
        gemini_status = f"❌ Error: {str(e)}"
    
    return openai_status, gemini_status

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

def verify_text(edited_text: str, original_text: str) -> bool:
    """Very lenient verification that edited text meets minimum quality standards."""
    if not edited_text or not original_text:
        return False
        
    # Extract quotes from both texts
    quote_pattern = r'"[^"]+"|«[^»]+»|"[^"]+"|\[[^\]]+\]'  # Added support for bracketed text
    original_quotes = set(re.findall(quote_pattern, original_text))
    edited_quotes = set(re.findall(quote_pattern, edited_text))
    
    # Very lenient quote preservation check - only check if there are quotes
    if original_quotes and len(original_quotes) > len(edited_quotes) * 0.3:  # Allow 70% of quotes to be modified
        logger.warning(f"Quote preservation ratio too low: {len(edited_quotes)}/{len(original_quotes)}")
        return False
    
    # Check for incomplete sentences (excluding titles and short phrases)
    lines = edited_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip titles, short lines, and special markers
        if len(line.split()) <= 7 or line.startswith('[') or line.endswith(']'):
            continue
            
        # Skip lines that are just quotes
        if re.match(quote_pattern, line):
            continue
            
        # Very lenient punctuation check - only for long lines
        if len(line.split()) > 15 and not any(line.endswith(p) for p in ['.', '!', '?', '؟', '۔', ':', '؛', '،', '-']):
            logger.warning(f"Long line missing punctuation: {line}")
            return False
    
    return True

def check_content_preserved(original_text: str, edited_text: str, threshold: float = 0.2) -> bool:
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
        return [s.strip() for s in re.split(r'[.!?؟۔।؛]', text) if s.strip()]
    
    original_sentences = get_sentences(original_cleaned)
    edited_sentences = get_sentences(edited_cleaned)
    
    # Check if all key concepts from original are present in edited
    original_words = set(w for s in original_sentences for w in s.split() if len(w) > 2)  # Only check words longer than 2 chars
    edited_words = set(w for s in edited_sentences for w in s.split() if len(w) > 2)
    
    # Calculate word preservation ratio with lower threshold
    preserved_words = len(original_words.intersection(edited_words))
    word_preservation_ratio = preserved_words / len(original_words) if original_words else 1.0
    
    # More lenient sentence count ratio
    sentence_ratio = len(edited_sentences) / len(original_sentences) if original_sentences else 1.0
    
    # Very lenient thresholds
    return word_preservation_ratio >= threshold and 0.3 <= sentence_ratio <= 1.7

def check_basic_completeness(text: str) -> bool:
    """Perform a basic check for text completeness."""
    if not text.strip():
        return False
        
    # Check if text has at least some complete sentences
    sentences = re.split(r'[.!?؟۔]', text)
    complete_sentences = 0
    for sentence in sentences:
        if len(sentence.strip().split()) > 1:  # Consider sentences with more than 1 word
            complete_sentences += 1
            
    return complete_sentences > 0  # At least one complete sentence

async def process_gemini_edit(text: str) -> str:
    """Process text editing using Gemini API with strict grammar and sentence completion."""
    if not text.strip():
        raise ValueError("Empty text provided")

    if len(text) > 60000:
        raise ValueError("Text too long for Gemini API (max 60000 characters)")

    # Pre-process text to identify titles, incomplete sentences and words
    def format_text_for_editing(text: str) -> str:
        # First identify titles (lines without ending punctuation that are followed by a blank line)
        lines = text.split('\n')
        formatted_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line:
                # Check if this line could be a title
                is_title = False
                if not any(line.endswith(p) for p in ['.', '!', '?', '؟', '۔']):
                    # Look ahead for blank line
                    if i + 1 < len(lines) and not lines[i + 1].strip():
                        is_title = True
                
                if is_title:
                    formatted_lines.append(f"[عنوان]: {line}")
                else:
                    # Handle regular sentences
                    if any(word.endswith('‌') for word in line.split()):
                        formatted_lines.append(f"[نیاز به تکمیل کلمات]: {line}")
                    elif not any(line.endswith(p) for p in ['.', '!', '?', '؟', '۔']):
                        formatted_lines.append(f"[نیاز به تکمیل جمله]: {line}")
                    else:
                        formatted_lines.append(line)
            else:
                formatted_lines.append(line)  # Keep blank lines
            i += 1
        return '\n'.join(formatted_lines)

    formatted_text = format_text_for_editing(text)
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            prompt = f"""لطفاً متن زیر را با دقت ویرایش کنید. به موارد زیر توجه ویژه نمایید:

۱. تشخیص و حفظ عنوان‌ها:
   - خطوط با علامت [عنوان] باید به عنوان عنوان حفظ شوند
   - عنوان‌ها نیازی به نقطه در پایان ندارند
   - فرمت و ساختار عنوان‌ها باید حفظ شود

۲. تکمیل کلمات و جملات:
   - عبارات با علامت [نیاز به تکمیل کلمات] دارای کلمات ناقص هستند که باید تکمیل شوند
   - عبارات با علامت [نیاز به تکمیل جمله] باید به جملات کامل تبدیل شوند
   - هر جمله (به جز عنوان‌ها) باید معنای کامل و مستقل داشته باشد

۳. حفظ محتوا:
   - هیچ جمله‌ای نباید حذف شود مگر با جایگزین مناسب
   - تمام مفاهیم اصلی باید در متن نهایی وجود داشته باشند
   - هر جمله اصلی باید حداقل یک معادل در متن ویرایش شده داشته باشد

۴. بهبود کیفیت:
   - استفاده از واژگان تخصصی و رسمی
   - اصلاح ساختار جملات
   - حفظ انسجام متن

متن اصلی برای ویرایش:
{formatted_text}

لطفاً متن را طوری ویرایش کنید که:
۱. عنوان‌ها به درستی شناسایی و حفظ شوند
۲. تمام کلمات ناقص تکمیل شوند
۳. هر جمله (به جز عنوان‌ها) کامل و معنادار باشد
۴. هیچ محتوایی بدون جایگزین حذف نشود
۵. سطح نگارش و واژگان ارتقا یابد"""

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
            
            # Verify no incomplete words in non-title lines
            lines = edited_text.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('[عنوان]'):
                    if any(word.endswith('‌') for word in line.split()):
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

def preserve_quotes(original_text: str, edited_text: str) -> str:
    """Preserve quotes from original text in edited text."""
    # Extract quotes from original text
    quote_pattern = r'"[^"]+"|«[^»]+»|"[^"]+"'
    original_quotes = re.findall(quote_pattern, original_text)
    edited_quotes = re.findall(quote_pattern, edited_text)
    
    # If edited text has fewer quotes, try to preserve original ones
    if len(edited_quotes) < len(original_quotes):
        for orig_quote in original_quotes:
            if orig_quote not in edited_text:
                # Find a suitable position to insert the quote
                sentences = re.split(r'([.!?؟۔])', edited_text)
                for i, sentence in enumerate(sentences):
                    if any(word in sentence for word in orig_quote.split()):
                        sentences[i] = f"{sentence} {orig_quote}"
                        break
                edited_text = ''.join(sentences)
    
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

async def process_openai_edit(text: str, model: str = "gpt-3.5-turbo") -> str:
    """Process text editing using OpenAI API with minimal changes."""
    if not text.strip():
        raise ValueError("Empty text provided")
            
    if len(text) > 50000:
        raise ValueError("Text too long for OpenAI API (max 50000 characters)")

    # Pre-process text to identify quotes and special sections
    def format_text_for_editing(text: str) -> str:
        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append(line)
                continue
                
            # Mark quotes
            quote_pattern = r'"[^"]+"|«[^»]+»|"[^"]+"'
            quotes = re.findall(quote_pattern, line)
            for i, quote in enumerate(quotes):
                line = line.replace(quote, f"[نقل قول]: {quote}")
            
            # Mark titles (lines without punctuation followed by blank line)
            if not any(line.endswith(p) for p in ['.', '!', '?', '؟', '۔']):
                formatted_lines.append(f"[عنوان]: {line}")
            else:
                formatted_lines.append(line)
                
        return '\n'.join(formatted_lines)

    formatted_text = format_text_for_editing(text)

    # Adjust parameters based on model
    if model == "gpt-4-turbo-preview":
        temperature = 0.3  # Slightly more creative
        max_tokens = min(4000, len(text.split()) * 2)
    else:  # gpt-3.5-turbo
        temperature = 0.4  # More creative
        max_tokens = min(3000, len(text.split()) * 2)

    # Simple system prompt
    system_prompt = """You are a Persian text editor. Your task is to improve the text while preserving its meaning:

1. Keep all text marked with [نقل قول] exactly as is
2. Keep all text marked with [عنوان] as titles
3. Fix grammar and punctuation errors
4. Improve sentence structure and flow
5. Keep the same meaning and key points

Most importantly:
- Make minimal necessary changes
- Never remove content without replacement
- Preserve all quotes and titles exactly"""

    # Simple user prompt
    user_prompt = f"""Edit this Persian text to improve its quality while keeping its meaning:

Text to edit:
{formatted_text}"""

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            top_p=0.7
        )
        
        if not response.choices:
            raise ValueError("No response from OpenAI API")
            
        edited_text = response.choices[0].message.content.strip()
        
        if not edited_text:
            raise ValueError("Empty response from OpenAI API")
        
        # Very lenient verification
        if not verify_text(edited_text, text):
            logger.warning("Text verification failed, attempting to fix...")
            edited_text = preserve_quotes(text, edited_text)
            
            if not verify_text(edited_text, text):
                logger.warning("Second verification failed, using more lenient check...")
                if not check_basic_completeness(edited_text):
                    logger.warning("Basic completeness check failed...")
                    if not check_content_preserved(text, edited_text, threshold=0.1):
                        raise ValueError("Failed to preserve content while editing")
        
        # Content preservation check with multiple thresholds
        if not check_content_preserved(text, edited_text, threshold=0.2):
            logger.warning("Content preservation check failed, attempting more lenient threshold...")
            if not check_content_preserved(text, edited_text, threshold=0.1):
                raise ValueError("Failed to preserve content while editing")
            
        return edited_text
        
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise ValueError(f"OpenAI API error: {str(e)}")
    except asyncio.TimeoutError:
        logger.error("OpenAI API timeout")
        raise TimeoutError("OpenAI API timeout")
    except Exception as e:
        logger.error(f"Error in OpenAI processing: {str(e)}")
        raise

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
    openai_status, gemini_status = await test_api_connections()
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
    openai_status, gemini_status = asyncio.run(test_api_connections())
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