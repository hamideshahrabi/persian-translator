from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Tuple, Any
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
import httpx
import aiohttp
import backoff  # For exponential backoff
from typing import Optional, Dict, Any
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from improvements import TRANSLATION_PROMPT, EDIT_PROMPT, EDIT_PROMPT_DETAILED
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import psutil
from openai import OpenAI, AsyncOpenAI

# Configure logging and load API keys
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables once at startup
load_dotenv(override=True)  # Force reload of environment variables

# Initialize API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize OpenAI with better configuration
@lru_cache()
def get_sync_openai_client():
    """Get or create synchronous OpenAI client instance."""
    try:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=httpx.Timeout(60.0, connect=10.0, read=30.0, write=30.0),
            max_retries=3
        )
    except Exception as e:
        logger.error(f"Error initializing sync OpenAI client: {str(e)}")
        raise

@lru_cache()
def get_async_openai_client():
    """Get or create async OpenAI client instance."""
    try:
        return AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=httpx.Timeout(60.0, connect=10.0, read=30.0, write=30.0),
            max_retries=3
        )
    except Exception as e:
        logger.error(f"Error initializing async OpenAI client: {str(e)}")
        raise

# Initialize models
@lru_cache()
def get_gemini_model(model_name: str = "models/gemini-1.5-pro-latest") -> genai.GenerativeModel:
    """Get or create Gemini model instance."""
    try:
        generation_config = {
            "temperature": 0.1,  # Lower temperature for faster, more deterministic responses
            "top_p": 0.95,      # Higher top_p for faster sampling
            "top_k": 20,        # Lower top_k for faster sampling
            "max_output_tokens": 2048,  # Increased for longer texts
            "candidate_count": 1,  # Only generate one candidate
        }
        
        logger.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        # Test the model with a simple prompt
        logger.info("Testing Gemini model with simple prompt...")
        response = model.generate_content("Test.")
        if not response or not response.text:
            logger.error("Failed to generate content with Gemini model")
            raise Exception("Failed to generate content with Gemini model")
            
        logger.info("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
        raise

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9090", "http://localhost:9091", "http://localhost:9092"],  # Add your production domains here
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Specify only the methods you need
    allow_headers=["Content-Type", "Authorization"],  # Specify only the headers you need
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define editing stages and context
EDIT_STAGES = {
    'grammar': {
        'name': 'اصلاح دستور زبان و نگارش',
        'focus': [
            'اصلاح خطاهای دستوری',
            'بهبود نقطه‌گذاری',
            'اصلاح فاصله‌گذاری'
        ],
        'examples': 'مثال: تبدیل "من رفتم خانه." به "من به خانه رفتم."'
    },
    'clarity': {
        'name': 'بهبود وضوح و روانی',
        'focus': [
            'بهبود ساختار جملات',
            'حذف ابهام',
            'افزایش روانی متن'
        ],
        'examples': 'مثال: ساده‌سازی جملات پیچیده با حفظ معنی'
    },
    'professional': {
        'name': 'استانداردسازی حرفه‌ای',
        'focus': [
            'یکدست‌سازی اصطلاحات',
            'حفظ لحن حرفه‌ای',
            'بهبود انسجام متن'
        ],
        'examples': 'مثال: استفاده از اصطلاحات استاندارد حرفه‌ای'
    }
}

# Global variables for API clients and keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Connection settings
MAX_RETRIES = 3
TIMEOUT = 30.0
BACKOFF_FACTOR = 2
INITIAL_BACKOFF = 1

# Validate API keys with proper error handling
def validate_api_keys():
    """Validate API keys and raise descriptive errors."""
    missing_keys = []
    if not OPENAI_API_KEY:
        missing_keys.append("OpenAI API key")
    if not GEMINI_API_KEY:
        missing_keys.append("Gemini API key")
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

# Initialize API clients with proper error handling
async def initialize_api_clients():
    """Initialize API clients and verify they work."""
    try:
        # Test OpenAI client
        client = get_sync_openai_client()
        models = client.models.list()  # Sync call for initialization
        logger.info("OpenAI client initialized successfully")
        logger.info(f"Available OpenAI models: {[model.id for model in models.data]}")

        # Test Gemini model
        model = get_gemini_model()
        logger.info("Gemini model initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize API clients: {str(e)}")
        return False

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GEMINI_FLASH = "models/gemini-1.5-flash-8b"
    GEMINI_PRO = "models/gemini-1.5-pro-latest"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.GPT35: "Fast and reliable processing with good accuracy",
            self.GPT4: "Most accurate processing, better understanding of context and nuances",
            self.GEMINI_FLASH: "Gemini 1.5 Flash - Fast and efficient model for text processing",
            self.GEMINI_PRO: "Gemini 1.5 Pro - Advanced model with better understanding"
        }
        return descriptions.get(self, "Unknown model")
        
    @property
    def max_tokens(self) -> int:
        limits = {
            self.GPT35: 30000,
            self.GPT4: 50000,
            self.GEMINI_FLASH: 1_000_000,
            self.GEMINI_PRO: 2000000
        }
        return limits.get(self, 30000)  # Default to 30000 if unknown

class EditMode(str, Enum):
    FAST = "fast"
    DETAILED = "detailed"

class EditRequest(BaseModel):
    text: str
    model: str
    mode: EditMode = EditMode.FAST

class EditResponse(BaseModel):
    edited_text: str
    technical_explanation: str

# Map frontend model names to backend model types
model_mapping = {
    "models/gemini-1.5-pro-latest": ModelType.GEMINI_PRO.value,
    "models/gemini-1.5-flash-8b": ModelType.GEMINI_FLASH.value,
    "gpt-3.5-turbo": ModelType.GPT35.value,
    "gpt-4": ModelType.GPT4.value,
    "gemini-pro": ModelType.GEMINI_PRO.value,  # For backward compatibility
    "gemini-flash": ModelType.GEMINI_FLASH.value  # For backward compatibility
}

def validate_word_count(original_text: str, edited_text: str, tolerance: int = 50) -> bool:
    """Validate that the edited text's word count is within tolerance of the original."""
    original_words = len(original_text.split())
    edited_words = len(edited_text.split())
    difference = abs(original_words - edited_words)
    return difference <= tolerance

async def process_gemini_edit(text: str, mode: EditMode = EditMode.FAST) -> str:
    """Process text editing using Gemini API with chunking for long texts."""
    try:
        logger.info("Starting Gemini edit process")
        model = get_gemini_model(ModelType.GEMINI_PRO.value)
        
        # Define the base prompt based on mode
        base_prompt = """شما یک ویراستار متخصص متون کوچینگ هستید. متن زیر را با حفظ معنا و کیفیت ویرایش کنید.

دستورالعمل‌های دقیق:
۱. متن را با دقت ویرایش کنید و تمام خطاها را اصلاح کنید
۲. تعداد کلمات متن ویرایش شده باید تقریباً برابر با متن اصلی باشد (حداکثر ۱۰۰ کلمه تفاوت)
۳. عناوین و سرفصل‌ها را دقیقاً حفظ کنید
۴. پاراگراف‌بندی را حفظ کنید
۵. اصلاحات ضروری را انجام دهید:
   - اصلاح خطاهای دستوری و املایی
   - بهبود ساختار جملات
   - حذف تکرار و حشو
   - بهینه‌سازی جملات
   - اصلاح خطاهای نگارشی
   - بهبود نقطه‌گذاری
   - اصلاح فاصله‌گذاری
   - بهبود روانی متن
   - اصلاح خطاهای منطقی
۶. لحن حرفه‌ای و رسمی را حفظ کنید
۷. اصطلاحات تخصصی کوچینگ را حفظ کنید
۸. معنا و محتوای اصلی را حفظ کنید
۹. از زبان رسمی و معیار استفاده کنید
۱۰. جملات را روان و واضح کنید
۱۱. حفظ پیوستگی و انسجام متن
۱۲. اصلاح خطاهای زمانی و مکانی در متن
۱۳. حفظ پیوستگی بین بخش‌های مختلف متن
۱۴. اطمینان از انسجام معنایی بین جملات

نکته مهم: هدف اصلی بهبود کیفیت متن و اصلاح خطاهاست. اگر خطایی در متن وجود دارد، حتماً آن را اصلاح کنید. در متون روایی و داستانی، حفظ روانی و انسجام متن از اهمیت بالایی برخوردار است. این متن بخشی از یک متن بزرگتر است، پس لطفاً پیوستگی معنایی و ساختاری آن را حفظ کنید."""
        
        # Split text into chunks if needed (max 400 words per chunk)
        words = text.split()
        chunks = []
        current_chunk = []
        current_word_count = 0
        max_words_per_chunk = 400
        
        for word in words:
            current_chunk.append(word)
            current_word_count += 1
            if current_word_count >= max_words_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        
        edited_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_prompt = f"{base_prompt}\n\nمتن برای ویرایش (بخش {i+1}/{len(chunks)}):\n{chunk}\n\nمتن ویرایش شده:"
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Attempt {attempt + 1}/{max_attempts}")
                    response = model.generate_content(chunk_prompt)
                
                    if not response or not response.text:
                        logger.warning(f"Empty response on attempt {attempt + 1}")
                        continue
                    
                    edited_chunk = response.text.strip()
                    
                    # Log the differences between original and edited text
                    original_words = len(chunk.split())
                    edited_words = len(edited_chunk.split())
                    logger.info(f"Original words: {original_words}, Edited words: {edited_words}")
                    
                    # Validate word count for this chunk with more flexible tolerance
                    if validate_word_count(chunk, edited_chunk, tolerance=100):
                        logger.info(f"Word count validation passed for chunk {i+1}")
                        edited_chunks.append(edited_chunk)
                        break
                    else:
                        logger.warning(f"Word count validation failed for chunk {i+1} on attempt {attempt + 1}")
                        
                    if attempt == max_attempts - 1:
                        logger.warning(f"All attempts failed for chunk {i+1}, using original chunk")
                        edited_chunks.append(chunk)
                
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_attempts - 1:
                        raise Exception(f"Failed to process chunk {i+1} after {max_attempts} attempts: {str(e)}")
                    continue
                
        # Combine edited chunks with proper spacing and ensure continuity
        combined_text = '\n\n'.join(edited_chunks)
        
        # Clean up any double newlines and ensure proper spacing
        combined_text = re.sub(r'\n\s*\n', '\n\n', combined_text)
        
        return combined_text
            
    except Exception as e:
        logger.error(f"Gemini edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini edit failed: {str(e)}")

async def chunk_text(text: str, max_chunk_size: int = 1500) -> List[str]:
    """Split text into chunks while preserving paragraph structure."""
    try:
        # First, normalize line breaks to ensure consistent paragraph separation
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
        
        # Split text into paragraphs while preserving empty lines
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                # If we have accumulated paragraphs, add them as a chunk first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append('')  # Add empty line
                continue
            
            # If a single paragraph is larger than max_chunk_size, split it into sentences
            if len(paragraph) > max_chunk_size:
                # If we have accumulated paragraphs, add them as a chunk first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split into sentences while preserving sentence endings
                sentences = []
                current_sentence = []
                words = paragraph.split()
                
                for word in words:
                    current_sentence.append(word)
                    if word.endswith(('.', '!', '?', '؛', '؟', '!', '،')):
                        sentences.append(' '.join(current_sentence))
                        current_sentence = []
                
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_size = len(sentence)
                    
                    # If sentence is too long, split into words
                    if sentence_size > max_chunk_size:
                        words = sentence.split()
                        word_chunk = []
                        word_size = 0
                        
                        for word in words:
                            word_size_with_space = len(word) + 1
                            
                            if word_size + word_size_with_space > max_chunk_size:
                                if word_chunk:
                                    chunks.append(' '.join(word_chunk))
                                word_chunk = [word]
                                word_size = len(word)
                            else:
                                word_chunk.append(word)
                                word_size += word_size_with_space
                        
                        if word_chunk:
                            chunks.append(' '.join(word_chunk))
                    else:
                        # Handle regular sentences
                        if temp_size + sentence_size > max_chunk_size and temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk = [sentence]
                            temp_size = sentence_size
                        else:
                            temp_chunk.append(sentence)
                            temp_size += sentence_size + 1  # Account for space
                
                # Handle remaining sentences
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            else:
                # Handle regular paragraphs
                if current_size + len(paragraph) > max_chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_size = len(paragraph)
                else:
                    current_chunk.append(paragraph)
                    current_size += len(paragraph) + 2  # Account for newline
        
        # Handle the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error in chunk_text: {str(e)}")
        raise

async def process_openai_edit(text: str, model: str = ModelType.GPT35.value, mode: EditMode = EditMode.FAST) -> str:
    """Process text editing using OpenAI API with chunking for long texts."""
    try:
        logger.info(f"Starting OpenAI edit process with model: {model}")
        client = get_async_openai_client()
        
        # Optimize chunk sizes for better performance
        max_chunk_size = 1500 if model == "gpt-4" else 1000  # Even smaller chunks for faster processing
        chunks = await chunk_text(text, max_chunk_size)  # Add await here
        logger.info(f"Split text into {len(chunks)} chunks")
        
        edited_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                # Different prompts for fast vs detailed editing
                if mode == EditMode.FAST:
                    system_prompt = """شما یک ویراستار متخصص متون کوچینگ در یک انتشارات حرفه‌ای هستید. هدف شما بهبود کیفیت متن با حفظ معنی، محتوای اصلی و اصطلاحات تخصصی کوچینگ است.
لطفاً متن را با رعایت موارد زیر ویرایش کنید:
۱. حفظ دقیق ساختار پاراگراف‌های متن اصلی
۲. حفظ تعداد پاراگراف‌ها و فاصله‌گذاری بین آنها
۳. اصلاح خطاهای دستوری و نقطه‌گذاری
۴. حفظ و استفاده صحیح از اصطلاحات تخصصی کوچینگ
۵. بهبود ساختار جملات و روانی متن
۶. حفظ دقیق معنا و محتوای اصلی
۷. رعایت اصول نگارش فارسی معیار
۸. تأکید بر حفظ و تقویت لحن رسمی و حرفه‌ای

نکته مهم: تعداد کلمات متن ویرایش شده باید تقریباً برابر با متن اصلی باشد (حداکثر ۲۰ کلمه کمتر یا بیشتر).
توجه: فقط ویرایش متن فارسی، بدون ترجمه."""
                else:
                    system_prompt = """شما یک ویراستار متخصص متون کوچینگ در یک انتشارات حرفه‌ای هستید. هدف شما بهبود کیفیت متن با حفظ معنی، محتوای اصلی و اصطلاحات تخصصی کوچینگ است.
لطفاً متن را با دقت و با رعایت موارد زیر ویرایش کنید:
۱. حفظ دقیق ساختار پاراگراف‌های متن اصلی
۲. حفظ تعداد پاراگراف‌ها و فاصله‌گذاری بین آنها
۳. اصلاح تمام خطاهای دستوری، املایی و نقطه‌گذاری
۴. حفظ و کاربرد دقیق اصطلاحات تخصصی کوچینگ
۵. بهبود ساختار جملات، وضوح و خوانایی متن
۶. ارتقای سطح نگارش و حرفه‌ای‌تر کردن متن
۷. حفظ دقیق معنا و محتوای اصلی
۸. رعایت اصول نگارش فارسی معیار و زبان رسمی
۹. تأکید ویژه بر حفظ و تقویت لحن رسمی و کاملاً حرفه‌ای

نکته مهم: تعداد کلمات متن ویرایش شده باید تقریباً برابر با متن اصلی باشد (حداکثر ۲۰ کلمه کمتر یا بیشتر).
توجه: فقط ویرایش متن فارسی، بدون ترجمه."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""لطفاً این متن کوچینگ (بخش {i+1}/{len(chunks)}) را با حفظ معنا، محتوا و اصطلاحات تخصصی آن ویرایش کنید.
تعداد کلمات متن ویرایش شده باید تقریباً برابر با متن اصلی باشد.

{chunk}

نکته مهم: فقط ویرایش متن فارسی، بدون ترجمه."""}
                ]
                
                max_attempts = 2  # Reduced from 3 to 2 attempts
                for attempt in range(max_attempts):
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,  # Lower temperature for faster, more consistent responses
                        max_tokens=max_chunk_size,
                        timeout=20.0,  # Further reduced timeout
                        presence_penalty=-0.1,  # Slight penalty to prevent wordiness
                        frequency_penalty=0.1,  # Slight penalty to prevent repetition
                    )
                    
                    if not completion or not completion.choices:
                        logger.error(f"Empty response from OpenAI for chunk {i+1} on attempt {attempt + 1}")
                        continue
                    
                    edited_chunk = completion.choices[0].message.content.strip()
                    
                    # Validate word count for this chunk
                    if validate_word_count(chunk, edited_chunk):
                        edited_chunks.append(edited_chunk)
                        break
                    else:
                        logger.warning(f"Word count validation failed for chunk {i+1} on attempt {attempt + 1}")
                        
                    if attempt == max_attempts - 1:
                        logger.warning(f"All attempts failed for chunk {i+1}, using original chunk")
                        edited_chunks.append(chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                edited_chunks.append(chunk)  # Keep original chunk if processing fails
                continue
        
        # Combine edited chunks with proper spacing
        edited_text = ''
        for i, chunk in enumerate(edited_chunks):
            chunk = chunk.strip()
            if not chunk:
                # Only add empty line if it was in the original text
                if i > 0 and i < len(edited_chunks) - 1 and edited_chunks[i-1].strip() and edited_chunks[i+1].strip():
                    edited_text += '\n\n'
                continue
            
            if i > 0 and edited_chunks[i-1].strip():
                # Add newline only between non-empty chunks
                edited_text += '\n\n'
            edited_text += chunk
        
        # Clean up any extra newlines while preserving paragraph structure
        edited_text = re.sub(r'\n\s*\n\s*\n', '\n\n', edited_text)
        edited_text = edited_text.strip()
        
        return edited_text
            
    except Exception as e:
        logger.error(f"OpenAI edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")

@app.post("/edit")
async def edit_text(request: EditRequest) -> EditResponse:
    """Edit Persian text for improved clarity and formality."""
    try:
        text = request.text.strip()
        model_type = model_mapping.get(request.model)
        mode = request.mode
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        if not model_type:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model}")
            
        # Update length validation for 3000 words
        max_length = 21000 if model_type == ModelType.GPT4.value else 15000  # Approximately 3000 words
        if len(text) > max_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long (max {max_length} characters, approximately 3000 words for {model_type})"
            )
            
        try:
            if model_type in [ModelType.GEMINI_FLASH.value, ModelType.GEMINI_PRO.value]:
                edited = await process_gemini_edit(text, mode)
            else:
                # Ensure we're using the correct model type for OpenAI
                if model_type == ModelType.GPT35.value:
                    model_type = "gpt-3.5-turbo-0125"  # Use the latest version
                elif model_type == ModelType.GPT4.value:
                    model_type = "gpt-4-0125-preview"  # Use the latest version
                edited = await process_openai_edit(text, model_type, mode)
            
            # Validate edited text
            if not edited or not edited.strip():
                raise Exception("Received empty response from API")
            
            # Create diff and ensure proper encoding
            diff_html = create_diff_html(text, edited)
            
            # Ensure proper encoding of Persian text
            return JSONResponse(
                content={
                    "edited_text": edited.strip(),
                    "technical_explanation": f"✅ Used {request.model} for {mode.value} editing",
                    "diff_html": diff_html,
                    "model_used": model_type
                },
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
        except Exception as e:
            logger.error(f"API processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process text with {request.model}: {str(e)}"
            )
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Edit error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during text editing: {str(e)}"
        )

def is_title(text: str, is_paragraph_start: bool = False) -> bool:
    """Check if a line is likely a title based on its characteristics and context."""
    # Remove extra whitespace
    text = text.strip()
    
    # Skip empty lines
    if not text:
        return False
        
    # If it's too long, it's probably not a title
    if len(text) > 100:
        return False
    
    # Check for section markers like "پرده اول", "پرده دوم", etc.
    section_markers = ['پرده', 'فصل', 'بخش', 'قسمت', 'گفتار']
    if any(text.startswith(marker) for marker in section_markers):
        return True
    
    # Common Persian verbs to check for their absence in potential titles
    persian_verbs = ['است', 'بود', 'شد', 'کرد', 'گفت', 'رفت', 'آمد', 'داد', 'دید', 'خواست',
                     'می‌شود', 'می‌کند', 'می‌گوید', 'می‌رود', 'می‌آید', 'می‌دهد', 'می‌بیند', 'می‌خواهد']
    
    # If it's a short phrase (1-8 words) and doesn't contain common verbs, it's likely a title
    words = text.split()
    if len(words) <= 8:
        # Check if the phrase has no common verbs
        has_verb = any(verb in text for verb in persian_verbs)
        if not has_verb:
            # If it's at the start of a paragraph, more likely to be a title
            if is_paragraph_start:
                return True
            
            # Check for common title indicators
            title_indicators = [':', '؛', '؟', '!', '.', '-', ')', '(']
            if any(indicator in text for indicator in title_indicators):
                return True
            
            # Check for Persian numbers at start
            persian_numbers = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
            if any(text.startswith(num) for num in persian_numbers):
                return True
            
            # Check for English numbers at start
            if re.match(r'^\d+[\.\-\)]\s+', text):
                return True
            
            # Check for common Persian title words
            title_words = ['مقدمه', 'نتیجه', 'درباره', 'موضوع', 'عنوان', 'داستان', 'حکایت']
            if any(word in text for word in title_words):
                return True
    
    return False

def create_diff_html(original: str, edited: str) -> str:
    """Create HTML with highlighted differences between original and edited text."""
    html = []
    # Track changes
    changes = {
        'replacements': 0,
        'deletions': 0,
        'insertions': 0
    }
    
    html.append("""
    <style>
        .diff-container {
            white-space: normal;
            word-wrap: break-word;
            font-family: inherit;
            line-height: 1.8;
            direction: rtl;
            text-align: right;
            padding: 1em;
            display: inline;
            width: 100%;
        }
        .word {
            display: inline;
            margin: 0 1px;
            white-space: normal;
            line-height: inherit;
        }
        .delete {
            color: #ff0000;
            background-color: #ffebee;
            padding: 2px;
            border-radius: 3px;
            display: inline;
            white-space: normal;
            line-height: inherit;
        }
        .insert {
            color: #008000;
            background-color: #e8f5e9;
            padding: 2px;
            border-radius: 3px;
            display: inline;
            white-space: normal;
            line-height: inherit;
        }
        .arrow {
            color: #666;
            margin: 0 1px;
            font-size: 0.9em;
            display: inline;
            white-space: normal;
            line-height: inherit;
        }
        .unchanged {
            display: inline;
            margin: 0 1px;
            white-space: normal;
            line-height: inherit;
        }
        .changes-summary {
            direction: rtl;
            text-align: right;
            margin-top: 1em;
            padding: 0.5em;
            background-color: #f5f5f5;
            border-radius: 4px;
            font-size: 0.9em;
            color: #666;
        }
        .word-count {
            color: #333;
            font-weight: bold;
        }
    </style>
    <div class="diff-container">""")

    # Split text into words while preserving spaces and punctuation
    def split_into_words(text):
        # Split by whitespace but keep the spaces
        parts = text.split()
        words = []
        for part in parts:
            # Split further by punctuation but keep the punctuation
            for word in re.finditer(r'[\w\u0600-\u06FF]+|[^\w\s\u0600-\u06FF]', part):
                words.append(word.group())
        return words

    # Get words from both texts
    orig_words = split_into_words(original)
    edit_words = split_into_words(edited)

    # Create word mapping using SequenceMatcher
    matcher = SequenceMatcher(None, orig_words, edit_words)

    # Process each operation from the matcher
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            # Words are identical
            for word in orig_words[i1:i2]:
                html.append(f'<span class="unchanged">{escape(word)} </span>')
        elif op == 'replace':
            # Words were changed
            html.append('<span class="word">')
            html.append(f'<span class="delete">{escape(" ".join(orig_words[i1:i2]))}</span>')
            html.append('<span class="arrow">→</span>')
            html.append(f'<span class="insert">{escape(" ".join(edit_words[j1:j2]))}</span>')
            html.append('</span> ')
            changes['replacements'] += 1
        elif op == 'delete':
            # Words were deleted
            html.append('<span class="word">')
            html.append(f'<span class="delete">{escape(" ".join(orig_words[i1:i2]))}</span>')
            html.append('</span> ')
            changes['deletions'] += 1
        elif op == 'insert':
            # Words were inserted
            html.append('<span class="word">')
            html.append(f'<span class="insert">{escape(" ".join(edit_words[j1:j2]))}</span>')
            html.append('</span> ')
            changes['insertions'] += 1

    # Calculate total word counts
    orig_word_count = len(original.split())
    edit_word_count = len(edited.split())
    word_diff = edit_word_count - orig_word_count

    # Add summary of changes
    total_changes = sum(changes.values())
    html.append('</div>')
    html.append(f"""
    <div class="changes-summary">
        خلاصه تغییرات:
        <br>
        📊 تعداد کلمات متن اصلی: <span class="word-count">{orig_word_count}</span>
        <br>
        📊 تعداد کلمات متن ویرایش شده: <span class="word-count">{edit_word_count}</span>
        <br>
        📊 تفاوت تعداد کلمات: <span class="word-count">{word_diff:+d}</span>
        <br>
        <br>
        🔄 تعداد کل تغییرات: {total_changes}
        <br>
        🔀 جایگزینی‌ها: {changes['replacements']}
        <br>
        ❌ حذف‌ها: {changes['deletions']}
        <br>
        ✅ اضافه‌ها: {changes['insertions']}
    </div>
    """)

    return '\n'.join(html)

@app.post("/translate")
async def translate_text(request: Request):
    """Translate Persian text to English with support for long texts."""
    try:
        data = await request.json()
        logger.info(f"Received translation request with data: {data}")
        
        text = data.get("text", "").strip()
        model_type = data.get("model", "gpt-3.5-turbo")
        
        logger.info(f"Starting translation with model: {model_type}")
        logger.info(f"Text length: {len(text)} characters")
        
        if not text:
            logger.error("Empty text provided")
            raise HTTPException(status_code=400, detail="No text provided")
            
        # Validate model type
        valid_models = ["gpt-3.5-turbo", "gpt-4", "models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-8b"]
        if model_type not in valid_models:
            logger.error(f"Invalid model type: {model_type}. Valid models are: {valid_models}")
            raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        
        # Process translation based on model
        try:
            if model_type in ["gpt-3.5-turbo", "gpt-4"]:
                logger.info(f"Using OpenAI model: {model_type}")
                try:
                    client = get_async_openai_client()
                    
                    # Split text into chunks if needed
                    max_chunk_size = 2500 if model_type == "gpt-4" else 1500  # Reduced chunk sizes for translation
                    chunks = await chunk_text(text, max_chunk_size)
                    logger.info(f"Split text into {len(chunks)} chunks")
                    
                    translated_chunks = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Translating chunk {i+1}/{len(chunks)}")
                        try:
                            completion = await client.chat.completions.create(
                                model=model_type,
                                messages=[
                                    {"role": "system", "content": """You are a Persian to English translator. Your task is to translate the text while:
1. Providing the translation as a single continuous paragraph
2. Not adding any line breaks or paragraph breaks
3. Using spaces between sentences
4. Translating the content accurately and naturally
5. Not preserving original formatting - everything should be in one flowing paragraph"""},
                                    {"role": "user", "content": f"""Translate this text (part {i+1}/{len(chunks)}) to English. 
IMPORTANT: Provide the translation as a single continuous paragraph without any line breaks.

{chunk}"""}
                                ],
                                temperature=0.3,
                                max_tokens=max_chunk_size,
                                timeout=60.0
                            )
                            
                            if not completion or not completion.choices:
                                logger.error(f"Empty response from OpenAI for chunk {i+1}")
                                raise HTTPException(status_code=500, detail=f"Empty response from OpenAI API for chunk {i+1}")
                            
                            chunk_translation = completion.choices[0].message.content.strip()
                            if not chunk_translation:
                                logger.error(f"Empty translation from OpenAI for chunk {i+1}")
                                raise HTTPException(status_code=500, detail=f"Empty translation from OpenAI for chunk {i+1}")
                            
                            translated_chunks.append(chunk_translation)
                            
                        except Exception as chunk_error:
                            logger.error(f"Error translating chunk {i+1}: {str(chunk_error)}")
                            raise HTTPException(status_code=500, detail=f"Error translating chunk {i+1}: {str(chunk_error)}")
                    
                    # Combine translated chunks into a single continuous paragraph
                    translated_text = ''
                    for chunk in translated_chunks:
                        chunk = chunk.strip()
                        if chunk:
                            # Remove any line breaks and extra spaces
                            chunk = chunk.replace('\n', ' ')
                            chunk = re.sub(r'\s+', ' ', chunk)
                            if translated_text:
                                translated_text += ' '
                            translated_text += chunk
                    
                    # Clean up any extra spaces and ensure single paragraph
                    translated_text = re.sub(r'\s+', ' ', translated_text)
                    translated_text = translated_text.strip()
                    
                    # Ensure no line breaks remain
                    translated_text = translated_text.replace('\n', ' ')
                    translated_text = re.sub(r'\s+', ' ', translated_text)
                    translated_text = translated_text.strip()
                    
                    # Add paragraph spacing based on original text structure
                    original_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    translated_paragraphs = []
                    current_pos = 0
                    
                    for orig_para in original_paragraphs:
                        # Find the corresponding translated text for this paragraph
                        orig_words = len(orig_para.split())
                        translated_words = translated_text.split()[current_pos:current_pos + orig_words]
                        translated_paragraphs.append(' '.join(translated_words))
                        current_pos += orig_words
                    
                    # Join paragraphs with double newlines
                    translated_text = '\n\n'.join(translated_paragraphs)
                    
                except openai.RateLimitError as e:
                    logger.error(f"OpenAI rate limit error: {str(e)}")
                    raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
                except openai.APITimeoutError as e:
                    logger.error(f"OpenAI timeout error: {str(e)}")
                    raise HTTPException(status_code=504, detail="Translation request timed out")
                except openai.APIError as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error with OpenAI: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
            
            elif model_type in ["models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-8b"]:
                logger.info("Using Gemini model")
                try:
                    model = get_gemini_model(model_type)
                    prompt = f"""Translate this Persian text to English accurately and naturally:

{text}

Instructions:
1. Translate the text while preserving the original paragraph structure
2. Keep each paragraph as a single continuous block of text
3. Use spaces between sentences within paragraphs
4. Add double line breaks between paragraphs
5. Translate accurately and naturally
6. Maintain the same number of paragraphs as the original text"""

                    response = model.generate_content(prompt)
                    if not response or not response.text:
                        logger.error("Empty response from Gemini API")
                        raise HTTPException(status_code=500, detail="Empty response from Gemini API")
                    translated_text = response.text.strip()
                    logger.info("Successfully got translation from Gemini")
                except Exception as e:
                    logger.error(f"Gemini API error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
            
            else:
                logger.error(f"Invalid model type: {model_type}")
                raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

            if not translated_text:
                logger.error("Empty translation result")
                raise HTTPException(status_code=500, detail="Empty translation result")

            logger.info("Successfully completed translation")
            return JSONResponse(
                content={
                    "translated_text": translated_text,
                    "model_used": model_type
                },
                headers={
                    "Content-Type": "application/json; charset=utf-8"
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Translation API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    """Serve the main page."""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request
        })
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import signal
    import sys
    import asyncio
    
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, cleaning up...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize API clients
        if not asyncio.run(initialize_api_clients()):
            logger.error("Failed to initialize API clients. Exiting...")
            sys.exit(1)
        
        # Try ports in a higher range
        start_port = 9090
        end_port = 9100
        
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
                
        for port in range(start_port, end_port):
            if is_port_in_use(port):
                logger.warning(f"Port {port} is already in use, trying next port...")
                continue
                
            try:
                logger.info(f"Starting server on port {port}")
                uvicorn.run(
                    "translation_bot:app",  # Pass as import string instead of app object
                    host="0.0.0.0",
                    port=port,
                    reload=True,
                    log_level="info",
                    access_log=True,
                    workers=1
                )
                break
            except Exception as e:
                logger.error(f"Failed to start server on port {port}: {str(e)}")
                if port == end_port - 1:
                    logger.error("No available ports found")
                    raise
                continue
                
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1) 