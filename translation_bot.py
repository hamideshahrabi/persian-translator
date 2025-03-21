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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

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
def get_gemini_model():
    """Get or create Gemini model instance."""
    try:
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-pro-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Test the model with a simple prompt
        response = model.generate_content("Test.")
        if not response or not response.text:
            raise Exception("Failed to generate content with Gemini model")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
        raise

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define editing stages and context
PUBLISHING_CONTEXT = """شما یک ویراستار متون فارسی در یک انتشارات حرفه‌ای هستید. هدف شما بهبود کیفیت متن با حفظ معنی و محتوای اصلی است."""

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
    GEMINI = "models/gemini-1.5-flash-8b"
    GEMINI2 = "models/gemini-1.5-pro-latest"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.GPT35: "Fast and reliable processing with good accuracy",
            self.GPT4: "Most accurate processing, better understanding of context and nuances",
            self.GEMINI: "Gemini 1.5 Flash - Fast and efficient model for text processing",
            self.GEMINI2: "Gemini 1.5 Pro - Advanced model with better understanding"
        }
        return descriptions.get(self, "Unknown model")
        
    @property
    def max_tokens(self) -> int:
        limits = {
            self.GPT35: 30000,
            self.GPT4: 50000,
            self.GEMINI: 1_000_000,
            self.GEMINI2: 2000000
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
    "models/gemini-1.5-pro-latest": ModelType.GEMINI2.value,
    "models/gemini-1.5-flash-8b": ModelType.GEMINI.value,
    "gpt-3.5-turbo": ModelType.GPT35.value,
    "gpt-4": ModelType.GPT4.value,
    "gemini-1.5-flash-8b": ModelType.GEMINI.value  # For translation section
}

async def process_gemini_edit(text: str, mode: EditMode = EditMode.FAST) -> str:
    """Process text editing using Gemini API."""
    try:
        logger.info("Starting Gemini edit process")
        model = get_gemini_model()
        
        prompt = f"""Edit this Persian text while maintaining ALL content and meaning:

        Text to edit: {text}
        
        Rules:
        1. Keep all main ideas and topics
        2. Improve grammar and style
        3. Use proper Persian punctuation and spacing
        
        Edited text:"""
        
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logger.warning("Empty response, returning original text")
            return text
            
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini edit error: {str(e)}")
        return text

def chunk_text(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into chunks while preserving paragraph structure."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start new one
        if current_size + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(paragraph)
        current_size += len(paragraph)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

async def process_openai_edit(text: str, model: str = ModelType.GPT35.value, mode: EditMode = EditMode.FAST) -> str:
    """Process text editing using OpenAI API with chunking for long texts."""
    try:
        logger.info(f"Starting OpenAI edit process with model: {model}")
        client = get_async_openai_client()
        
        # Optimize chunk sizes for better performance
        max_chunk_size = 3000 if model == "gpt-4" else 2000  # Smaller chunks for faster processing
        chunks = chunk_text(text, max_chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        edited_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                # Different prompts for fast vs detailed editing
                if mode == EditMode.FAST:
                    system_prompt = """You are a professional Persian text editor. Your task is to edit and improve the text while:
1. Fixing grammar and punctuation errors
2. Improving sentence structure and flow
3. Maintaining the exact same meaning and content
4. Using proper Persian writing standards
DO NOT translate the text. Only edit and improve the Persian text."""
                else:
                    system_prompt = """You are a professional Persian text editor. Your task is to carefully edit and improve the text while:
1. Fixing all grammar, spelling, and punctuation errors
2. Improving sentence structure, clarity, and readability
3. Enhancing the overall writing style and professionalism
4. Maintaining the exact same meaning and content
5. Using proper Persian writing standards and formal language
DO NOT translate the text. Only edit and improve the Persian text."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Edit this Persian text (part {i+1}/{len(chunks)}) to improve its quality while keeping the exact same meaning and content:

{chunk}

Important: DO NOT translate to English. Only edit and improve the Persian text."""}
                ]
                
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent editing
                    max_tokens=max_chunk_size,
                    timeout=45.0  # Reduced timeout for better performance
                )
                
                if not completion or not completion.choices:
                    logger.error(f"Empty response from OpenAI for chunk {i+1}")
                    edited_chunks.append(chunk)  # Keep original chunk if processing fails
                    continue
                    
                edited_chunk = completion.choices[0].message.content.strip()
                if not edited_chunk:
                    logger.error(f"Empty edited text from OpenAI for chunk {i+1}")
                    edited_chunks.append(chunk)
                    continue
                    
                edited_chunks.append(edited_chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                edited_chunks.append(chunk)  # Keep original chunk if processing fails
                continue
        
        # Combine edited chunks with proper spacing
        return '\n\n'.join(edited_chunks)
            
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
            
        # Add length validation with more reasonable limits
        max_length = 12000 if model_type == "gpt-4" else 8000  # Adjusted limits
        if len(text) > max_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long (max {max_length} characters for {model_type})"
            )
            
        try:
            if model_type in [ModelType.GEMINI.value, ModelType.GEMINI2.value]:
                edited = await process_gemini_edit(text, mode)
            else:
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

def create_diff_html(original: str, edited: str) -> str:
    """Create HTML with highlighted differences between original and edited text."""
    from difflib import SequenceMatcher
    
    def escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    matcher = SequenceMatcher(None, original, edited)
    html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Keep unchanged text as is
            html.append(escape_html(original[i1:i2]))
        elif tag == "delete":
            # Show deleted text with strikethrough and red color
            html.append(f'<span class="delete">{escape_html(original[i1:i2])}</span>')
        elif tag == "insert":
            # Show new text in green
            html.append(f'<span class="insert">{escape_html(edited[j1:j2])}</span>')
        elif tag == "replace":
            # Show both old (strikethrough) and new text side by side
            html.append(f'<span class="delete">{escape_html(original[i1:i2])}</span>')
            html.append('<span class="arrow">➜</span>')  # Add an arrow to show the change
            html.append(f'<span class="insert">{escape_html(edited[j1:j2])}</span>')
    
    return "".join(html)

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
        valid_models = ["gpt-3.5-turbo", "gpt-4", "gemini-pro"]
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
                    max_chunk_size = 4000 if model_type == "gpt-4" else 2500
                    chunks = chunk_text(text, max_chunk_size)
                    logger.info(f"Split text into {len(chunks)} chunks")
                    
                    translated_chunks = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Translating chunk {i+1}/{len(chunks)}")
                        completion = await client.chat.completions.create(
                            model=model_type,
                            messages=[
                                {"role": "system", "content": "You are a Persian to English translator. Translate the following text to English accurately and naturally."},
                                {"role": "user", "content": f"Translate this text (part {i+1}/{len(chunks)}):\n{chunk}"}
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
                    
                    translated_text = '\n\n'.join(translated_chunks)
                    logger.info("Successfully completed all translations")
                    
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
            
            elif model_type == "gemini-pro":
                logger.info("Using Gemini model")
                try:
                    model = get_gemini_model()
                    prompt = f"""Translate this Persian text to English accurately and naturally:

{text}

Instructions:
1. Maintain the original meaning and tone
2. Use natural English expressions
3. Keep any technical terms accurate
4. Preserve formatting and paragraph structure"""

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