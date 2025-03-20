from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Configure logging and load API keys
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables once at startup
load_dotenv(override=True)  # Force reload of environment variables

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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
openai_client = None
genai_initialized = False

@dataclass
class ConnectionState:
    is_connected: bool = False
    last_check: datetime = datetime.min
    error_count: int = 0
    last_error: Optional[str] = None
    retry_after: Optional[datetime] = None

class ConnectionPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._openai_state = ConnectionState()
        self._gemini_state = ConnectionState()
        self._env_loaded = False
        self._check_interval = timedelta(seconds=30)
        self._max_retries = 3
        self._retry_delay = 2  # seconds
        self._backoff_factor = 2  # exponential backoff factor
        self._openai_client = None
        self._gemini_model = None
        self._initialized = False
        self._last_error = None
        self._error_count = 0
        self._connection_timeout = 10.0  # seconds
        
    def _load_environment(self) -> bool:
        """Load environment variables with validation"""
        with self._lock:
            if self._env_loaded:
                return True
                
            try:
                # Force reload environment variables
                load_dotenv(override=True)
                
                # Validate required variables
                required_vars = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
                missing_vars = [var for var in required_vars if not os.getenv(var)]
                
                if missing_vars:
                    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    return False
                    
                # Validate API key formats
                openai_key = os.getenv('OPENAI_API_KEY')
                if not openai_key.startswith('sk-'):
                    error_msg = "OpenAI API key format appears incorrect. Expected to start with 'sk-'"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    return False
                    
                gemini_key = os.getenv('GEMINI_API_KEY')
                if not gemini_key.startswith('AIza'):
                    error_msg = "Gemini API key format appears incorrect. Expected to start with 'AIza'"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    return False
                    
                self._env_loaded = True
                self._last_error = None
                return True
            except Exception as e:
                error_msg = f"Failed to load environment: {str(e)}"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
                
    async def _check_openai_connection(self) -> bool:
        """Check OpenAI connection health"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                error_msg = "OpenAI API key not found"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
                
            logger.info(f"Attempting to initialize OpenAI client with API key (masked): {api_key[:8]}...")
            
            if not self._openai_client:
                self._openai_client = openai.AsyncOpenAI(
                    api_key=api_key,
                    timeout=self._connection_timeout
                )
            
            logger.info("Making test request to OpenAI API...")
            try:
                response = await self._openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                
                if response and response.choices:
                    logger.info("OpenAI test request successful")
                    self._last_error = None
                    return True
                else:
                    error_msg = "OpenAI API returned empty response"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    return False
                    
            except openai.AuthenticationError as e:
                error_msg = f"OpenAI authentication failed: {str(e)}"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
            except openai.RateLimitError as e:
                error_msg = f"OpenAI rate limit exceeded: {str(e)}"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
            except openai.APIError as e:
                error_msg = f"OpenAI API error: {str(e)}"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
                
        except Exception as e:
            error_msg = f"OpenAI connection check failed: {str(e)}"
            logger.error(error_msg)
            self._last_error = error_msg
            return False
            
    async def _check_gemini_connection(self) -> bool:
        """Check if Gemini API is accessible and working."""
        try:
            # Get and validate API key
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                error_msg = "Gemini API key not found in environment variables"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
            
            # Configure Gemini client
            logger.info(f"Configuring Gemini with API key (masked): {api_key[:8]}...")
            genai.configure(api_key=api_key)
            
            # Try to list available models first
            try:
                logger.info("Attempting to list available models")
                models = genai.list_models()
                
                if not models:
                    error_msg = "No models returned from Gemini API"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    return False
                    
                logger.info("Available models:")
                for m in models:
                    logger.info(f"Model: {m.name}")
                    logger.info(f"Generation methods: {m.supported_generation_methods}")
                    
                    # Try to use each model that supports text generation
                    if 'generateContent' in m.supported_generation_methods:
                        try:
                            logger.info(f"Attempting to use model: {m.name}")
                            model = genai.GenerativeModel(m.name)
                            response = model.generate_content("Test")
                            if response and response.text:
                                logger.info(f"Successfully connected to Gemini API using model: {m.name}")
                                self._gemini_model = model
                                self._last_error = None
                                return True
                        except Exception as model_error:
                            logger.error(f"Failed to use model {m.name}: {str(model_error)}")
                            continue
                
                error_msg = "No suitable Gemini model found"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
                
            except Exception as list_error:
                error_msg = f"Failed to list models: {str(list_error)}"
                logger.error(error_msg)
                self._last_error = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Gemini connection failed: {str(e)}"
            logger.error(error_msg)
            if "PERMISSION_DENIED" in str(e):
                error_msg = "This might be due to an invalid API key or insufficient permissions"
            elif "NOT_FOUND" in str(e):
                error_msg = "The specified model was not found. Please check the model name"
            elif "API key not valid" in str(e):
                error_msg = "The API key appears to be invalid. Please check your API key"
            elif "quota" in str(e).lower():
                error_msg = "API quota exceeded or billing not enabled"
            else:
                error_msg = f"Unexpected error: {str(e)}"
            self._last_error = error_msg
            return False
            
    async def _wait_for_retry(self, state: ConnectionState) -> bool:
        """Check if we should wait before retrying"""
        if state.retry_after and datetime.now() < state.retry_after:
            wait_time = (state.retry_after - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                await asyncio.sleep(wait_time)
            return True
        return False
        
    async def _handle_error(self, state: ConnectionState, error: Exception):
        """Handle connection errors with exponential backoff"""
        state.error_count += 1
        state.last_error = str(error)
        
        # Calculate next retry time with exponential backoff
        delay = self._retry_delay * (self._backoff_factor ** (state.error_count - 1))
        state.retry_after = datetime.now() + timedelta(seconds=delay)
        
        logger.warning(f"Connection error occurred. Will retry in {delay:.1f} seconds. Error: {str(error)}")
        
        # Reset error count after a while to prevent infinite backoff
        if state.error_count > 10:
            state.error_count = 0
            logger.info("Reset error count after max retries")
            
    async def ensure_connections(self) -> bool:
        """Ensure all connections are healthy with proper error handling"""
        if not self._load_environment():
            return False
            
        now = datetime.now()
        success = True
        
        # Check OpenAI connection if needed
        if (now - self._openai_state.last_check) > self._check_interval:
            with self._lock:
                if await self._wait_for_retry(self._openai_state):
                    return False
                    
                try:
                    self._openai_state.is_connected = await self._check_openai_connection()
                    if self._openai_state.is_connected:
                        self._openai_state.error_count = 0
                        self._openai_state.last_error = None
                    self._openai_state.last_check = now
                except Exception as e:
                    await self._handle_error(self._openai_state, e)
                    success = False
                    
        # Check Gemini connection if needed
        if (now - self._gemini_state.last_check) > self._check_interval:
            with self._lock:
                if await self._wait_for_retry(self._gemini_state):
                    return False
                    
                try:
                    self._gemini_state.is_connected = await self._check_gemini_connection()
                    if self._gemini_state.is_connected:
                        self._gemini_state.error_count = 0
                        self._gemini_state.last_error = None
                    self._gemini_state.last_check = now
                except Exception as e:
                    await self._handle_error(self._gemini_state, e)
                    success = False
                    
        return success
        
    def get_connection(self, service: str) -> Any:
        """Get a connection from the pool"""
        try:
            if service == "openai":
                return self._openai_client
            elif service == "gemini":
                return self._gemini_model
            else:
                raise ValueError(f"Unknown service: {service}")
        except Exception as e:
            logger.error(f"Failed to get connection for {service}: {str(e)}")
            raise
            
    async def close_all(self):
        """Close all connections"""
        try:
            # Close OpenAI connection if active
            if self._openai_client:
                await self._openai_client.close()
                self._openai_client = None
                self._openai_state = ConnectionState()
                
            # Reset Gemini state
            self._gemini_model = None
            self._gemini_state = ConnectionState()
            
            logger.info("All connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            raise

# Create global connection pool
connection_pool = ConnectionPool()

@asynccontextmanager
async def get_api_connection(service: str):
    """Context manager for API connections"""
    try:
        # Get connection from pool
        connection = connection_pool.get_connection(service)
        
        # For OpenAI, we need to ensure the connection is properly initialized
        if service == 'openai':
            if not connection:
                connection = openai.AsyncOpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=30.0,
                    max_retries=2
                )
                connection_pool._openai_client = connection
            
            # Test the connection
            try:
                await connection.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
            except Exception as e:
                logger.warning(f"OpenAI connection test failed, reinitializing: {str(e)}")
                connection = openai.AsyncOpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    timeout=30.0,
                    max_retries=2
                )
                connection_pool._openai_client = connection
            
        # For Gemini, we need to ensure the model is properly initialized
        elif service == 'gemini':
            if not connection:
                # Configure Gemini with API key
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError("Gemini API key not found")
                    
                genai.configure(api_key=api_key)
                
                # Try to use gemini-1.5-flash-8b model directly
                try:
                    logger.info("Attempting to use gemini-1.5-flash-8b model")
                    connection = genai.GenerativeModel('gemini-1.5-flash-8b')
                    # Test the model
                    response = connection.generate_content("Test message")
                    if response and response.text:
                        logger.info("Successfully initialized gemini-1.5-flash-8b model")
                        connection_pool._gemini_model = connection
                    else:
                        raise ValueError("gemini-1.5-flash-8b model returned empty response")
                except Exception as e:
                    logger.warning(f"Failed to use gemini-1.5-flash-8b model: {str(e)}")
                    
                    # Fallback: try to list available models
                    try:
                        logger.info("Attempting to list available models")
                        models = genai.list_models()
                        
                        if not models:
                            raise ValueError("No models returned from Gemini API")
                            
                        logger.info("Available models:")
                        for m in models:
                            logger.info(f"Model: {m.name}")
                            logger.info(f"Generation methods: {m.supported_generation_methods}")
                            
                            # Try to use each model that supports text generation
                            if 'generateContent' in m.supported_generation_methods:
                                try:
                                    logger.info(f"Attempting to use model: {m.name}")
                                    model = genai.GenerativeModel(m.name)
                                    response = model.generate_content("Test")
                                    if response and response.text:
                                        logger.info(f"Successfully connected to Gemini API using model: {m.name}")
                                        connection_pool._gemini_model = model
                                        connection = model  # Update the connection variable
                                        break
                                except Exception as model_error:
                                    logger.error(f"Failed to use model {m.name}: {str(model_error)}")
                                    continue
                        
                        if not connection:
                            raise ValueError("No suitable Gemini model found")
                        
                    except Exception as list_error:
                        logger.error(f"Failed to list models: {str(list_error)}")
                        raise
                    
        yield connection
        
    except Exception as e:
        logger.error(f"Error getting connection for {service}: {str(e)}")
        raise
    finally:
        # Only close OpenAI connections
        if service == 'openai' and connection:
            await connection.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        # Initialize connections
        if not await connection_pool.ensure_connections():
            raise RuntimeError("Failed to establish initial connections")
        logger.info("All connections initialized successfully")
        yield
    finally:
        await connection_pool.close_all()

# Setup API with lifespan
app = FastAPI(lifespan=lifespan)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Update the test_api_connections function to use the connection manager
async def test_api_connections():
    """Test API connections with detailed logging and automatic retries."""
    openai_status = "Not tested"
    gemini_status = "Not tested"
    gemini2_status = "Not tested"
    
    # Test OpenAI with retries
    try:
        async with get_api_connection('openai') as openai_client:
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            if response:
                logger.info("✅ Successfully tested GPT-3.5 connection")
                try:
                    if OPENAI_API_KEY.startswith("sk-org-"):
                        response = await openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": "Test"}],
                            max_tokens=5
                        )
                        openai_status = "✅ Connected (GPT-3.5 & GPT-4)"
                    else:
                        openai_status = "✅ Connected (GPT-3.5 only, GPT-4 requires Organization API key)"
                except Exception as e:
                    openai_status = "✅ Connected (GPT-3.5 only)"
            else:
                openai_status = "❌ Empty response"
    except Exception as e:
        logger.error(f"❌ OpenAI connection error: {str(e)}")
        openai_status = f"❌ Error: {str(e)}"
    
    # Test Gemini with retries
    try:
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found")
            
        genai.configure(api_key=api_key)
        
        # First try to use gemini-1.5-flash-8b model directly
        try:
            logger.info("Attempting to use gemini-1.5-flash-8b model")
            model = genai.GenerativeModel('gemini-1.5-flash-8b')
            response = model.generate_content("Test")
            if response and response.text:
                gemini_status = "✅ Connected"
                logger.info("Successfully connected to gemini-1.5-flash-8b model")
            else:
                raise ValueError("Empty response from gemini-1.5-flash-8b model")
        except Exception as e:
            logger.warning(f"Failed to use gemini-1.5-flash-8b model: {str(e)}")
            
            # Fallback: try to list available models
            logger.info("Attempting to list available models")
            models = genai.list_models()
            
            if not models:
                raise ValueError("No models returned from Gemini API")
                
            logger.info("Available Gemini models:")
            for m in models:
                logger.info(f"Model: {m.name}")
                logger.info(f"Generation methods: {m.supported_generation_methods}")
                logger.info(f"Display name: {m.display_name}")
                logger.info(f"Description: {m.description}")
                logger.info("---")
            
            # Try to find a suitable model
            for m in models:
                if "generateContent" in m.supported_generation_methods:
                    try:
                        model = genai.GenerativeModel(m.name)
                        response = model.generate_content("Test")
                        if response and response.text:
                            gemini_status = f"✅ Connected (using {m.name})"
                            logger.info(f"Successfully connected to model: {m.name}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to initialize model {m.name}: {str(e)}")
                        continue
            
            if gemini_status == "Not tested":
                gemini_status = "❌ No suitable Gemini model found"
                
        except Exception as e:
            logger.error(f"❌ Gemini connection error: {str(e)}")
            gemini_status = f"❌ Error: {str(e)}"
            
    except Exception as e:
        logger.error(f"❌ Gemini connection error: {str(e)}")
        gemini_status = f"❌ Error: {str(e)}"
        gemini2_status = f"❌ Error: {str(e)}"

    return openai_status, gemini_status, gemini2_status

class ModelType(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GEMINI = "gemini-1.5-flash-8b"
    GEMINI2 = "gemini-1.5-pro"
    
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
    status: str
    services: Dict[str, Dict[str, Optional[Union[str, bool]]]]
    message: str

def verify_text(edited_text: str, original_text: str) -> bool:
    """Strict verification that edited text meets professional quality standards."""
    if not edited_text or not original_text:
        return False
        
    # Extract quotes from both texts
    quote_pattern = r'"[^"]+"|«[^»]+»|"[^"]+"|\[[^\]]+\]'
    original_quotes = set(re.findall(quote_pattern, original_text))
    edited_quotes = set(re.findall(quote_pattern, edited_text))
    
    # Stricter quote preservation - only allow 10% modification
    if original_quotes and len(edited_quotes) < len(original_quotes) * 0.9:
        logger.warning(f"Quote preservation ratio too low: {len(edited_quotes)}/{len(original_quotes)}")
        return False
    
    # Check all sentences for proper punctuation
    lines = edited_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip only titles and special markers
        if line.startswith('[') or line.endswith(']'):
            continue
            
        # Skip lines that are just quotes
        if re.match(quote_pattern, line):
            continue
            
        # Check punctuation for all sentences
        if not any(line.endswith(p) for p in ['.', '!', '?', '؟', '۔', ':', '؛', '،', '-']):
            logger.warning(f"Sentence missing punctuation: {line}")
            return False
            
        # Additional professional content checks
        if len(line.split()) > 0:  # Check all non-empty lines
            # Check for proper spacing around punctuation
            if re.search(r'[.!?؟۔؛،][^\s]', line):
                logger.warning(f"Missing space after punctuation: {line}")
                return False
                
            # Check for professional formatting
            if re.search(r'\s{2,}', line):  # Multiple spaces
                logger.warning(f"Multiple spaces found: {line}")
                return False
    
    return True

def check_content_preserved(original_text: str, edited_text: str, threshold: float = 0.5) -> bool:
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
    
    # Calculate word preservation ratio with higher threshold
    preserved_words = len(original_words.intersection(edited_words))
    word_preservation_ratio = preserved_words / len(original_words) if original_words else 1.0
    
    # More strict sentence count ratio
    sentence_ratio = len(edited_sentences) / len(original_sentences) if original_sentences else 1.0
    
    # Stricter thresholds
    return word_preservation_ratio >= threshold and 0.7 <= sentence_ratio <= 1.3

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
    """Process text editing using Gemini API."""
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Try to use the specified model directly
        try:
            logger.info("Attempting to use gemini-1.5-flash-8b model")
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            
            # Create a proper translation prompt
            prompt = EDIT_PROMPT.format(text=text)
            
            # Generate translation with retry logic
            @backoff.on_exception(backoff.expo, Exception, max_tries=3)
            def generate_with_retry():
                return model.generate_content(prompt)
            
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, generate_with_retry),
                timeout=30.0
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")
                
            logger.info("Successfully generated translation")
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to use gemini-1.5-flash-8b model: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Gemini translation error: {str(e)}")
        raise

async def process_openai_edit(text: str, model: str = ModelType.GPT35.value) -> str:
    """Process text editing using OpenAI API."""
    try:
        # Create a proper translation prompt
        prompt = EDIT_PROMPT.format(text=text)
        
        # Generate translation with retry logic
        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        def generate_with_retry():
            return openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
        
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, generate_with_retry),
            timeout=30.0
        )
        
        if not response or not response.choices:
            raise ValueError("Empty response from OpenAI API")
            
        logger.info("Successfully generated translation")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {str(e)}")
        raise

def format_text_for_editing(text: str) -> str:
    """Format text for editing by adding clear instructions."""
    return f"""Please edit the following Persian text to improve its grammar, style, and clarity while maintaining its meaning. 
    Keep the same tone and formality level. Only make necessary changes.

    Original text:
    {text}

    Edited text:"""

@app.get("/")
async def read_root(request: Request):
    try:
        # Get current API status
        status = await get_status()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "api_status": status
        })
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        # Return a basic template even if status check fails
        return templates.TemplateResponse("index.html", {
            "request": request,
            "api_status": {
                "status": "error",
                "services": {
                    "openai": {"status": "unknown"},
                    "gemini": {"status": "unknown"}
                },
                "message": "Unable to check service status"
            }
        })

@app.post("/edit")
async def edit(request: EditRequest) -> EditResponse:
    """Edit Persian text using the specified model."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
            
        edited_text = text
        explanation = ""
        error_occurred = False
        changes = []
        diff_html = text
        
        try:
            logger.info(f"Received edit request with model: {request.model}")
            
            # Validate model type
            if request.model not in [m.value for m in ModelType]:
                logger.error(f"Invalid model type received: {request.model}")
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
            
            # Check connection state before processing
            if not await connection_pool.ensure_connections():
                logger.error("Failed to ensure API connections")
                raise HTTPException(status_code=503, detail="API services are currently unavailable. Please try again later.")
            
            if request.model in [ModelType.GEMINI.value, ModelType.GEMINI2.value]:
                if not connection_pool._gemini_state.is_connected:
                    logger.error("Gemini API not connected")
                    raise HTTPException(status_code=503, detail="Gemini API is currently unavailable. Please try again later or use a different model.")
                logger.info("Attempting to edit with Gemini model")
                try:
                    edited_text = await process_gemini_edit(text)
                    explanation = "✅ Used Gemini for editing"
                except Exception as e:
                    logger.error(f"Gemini processing error: {str(e)}")
                    if "PERMISSION_DENIED" in str(e):
                        raise HTTPException(status_code=401, detail="Gemini API authentication failed. Please check your API key.")
                    elif "NOT_FOUND" in str(e):
                        raise HTTPException(status_code=404, detail="Gemini model not found. Please try a different model.")
                    elif "quota" in str(e).lower():
                        raise HTTPException(status_code=429, detail="Gemini API quota exceeded. Please try again later.")
                    else:
                        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")
                    
            elif request.model in [ModelType.GPT35.value, ModelType.GPT4.value]:
                if not connection_pool._openai_state.is_connected:
                    logger.error("OpenAI API not connected")
                    raise HTTPException(status_code=503, detail="OpenAI API is currently unavailable. Please try again later or use a different model.")
                try:
                    if request.model == ModelType.GPT4.value and not OPENAI_API_KEY.startswith("sk-org-"):
                        logger.warning("GPT-4 requested but not available, falling back to GPT-3.5")
                        edited_text = await process_openai_edit(text, model=ModelType.GPT35.value)
                        explanation = "⚠️ GPT-4 not available, used GPT-3.5 instead"
                    else:
                        edited_text = await process_openai_edit(text, model=request.model)
                        explanation = f"✅ Used {request.model} for editing"
                except Exception as e:
                    logger.error(f"OpenAI processing error: {str(e)}")
                    if "authentication" in str(e).lower():
                        raise HTTPException(status_code=401, detail="OpenAI API authentication failed. Please check your API key.")
                    elif "rate_limit" in str(e).lower():
                        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please wait a moment and try again.")
                    elif "context_length" in str(e).lower():
                        raise HTTPException(status_code=413, detail="Text is too long for the selected model. Please try with a shorter text.")
                    else:
                        raise HTTPException(status_code=500, detail=f"Error processing with OpenAI: {str(e)}")
            
        except HTTPException as e:
            logger.error(f"HTTP error in edit endpoint: {str(e)}")
            raise e
        except Exception as e:
            error_occurred = True
            error_msg = str(e)
            logger.error(f"Edit error with {request.model}: {error_msg}")
            
            if "rate_limit" in error_msg.lower():
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment and try again.")
            elif "timeout" in error_msg.lower():
                raise HTTPException(status_code=408, detail="Request timed out. Please try again with a shorter text or use a different model.")
            elif "api key" in error_msg.lower():
                raise HTTPException(status_code=401, detail="API authentication failed. Please contact support.")
            elif "context_length" in error_msg.lower():
                raise HTTPException(status_code=413, detail="Text is too long for the selected model. Please try with a shorter text.")
            else:
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {error_msg}")
            
        if edited_text == text and not error_occurred:
            explanation = "⚠️ No changes were made to the text. The model either found no improvements needed or failed to make meaningful changes."
        elif edited_text != text:
            explanation = "✅ Successfully applied text improvements"
            changes = detect_changes(text, edited_text)
            diff_html = generate_diff_html(text, edited_text)
            logger.info(f"Successfully processed edit request. Changes detected: {len(changes)}")
            
        return EditResponse(
            edited_text=edited_text,
            technical_explanation=explanation,
            changes=changes,
            diff_html=diff_html
        )
            
    except HTTPException as e:
        logger.error(f"HTTP error in edit endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/status")
async def get_status() -> StatusResponse:
    """Get the current status of API services."""
    try:
        # Initialize connections if needed
        if not await connection_pool.ensure_connections():
            logger.warning("Failed to ensure connections")
            return StatusResponse(
                status="degraded",
                services={
                    "gemini": {
                        "status": "offline",
                        "last_checked": None,
                        "error": connection_pool._last_error or "Connection check failed"
                    },
                    "openai": {
                        "status": "offline",
                        "last_checked": None,
                        "error": connection_pool._last_error or "Connection check failed",
                        "gpt4_available": False
                    }
                },
                message="Some services are experiencing issues"
            )
        
        status = {
            "gemini": {
                "status": "online" if connection_pool._gemini_state.is_connected else "offline",
                "last_checked": connection_pool._gemini_state.last_check.isoformat() if connection_pool._gemini_state.last_check else None,
                "error": connection_pool._gemini_state.last_error or connection_pool._last_error
            },
            "openai": {
                "status": "online" if connection_pool._openai_state.is_connected else "offline",
                "last_checked": connection_pool._openai_state.last_check.isoformat() if connection_pool._openai_state.last_check else None,
                "error": connection_pool._openai_state.last_error or connection_pool._last_error,
                "gpt4_available": OPENAI_API_KEY.startswith("sk-org-")
            }
        }
        
        all_services_online = all(
            service["status"] == "online" 
            for service in status.values()
        )
        
        error_message = "All services operational" if all_services_online else "Some services are experiencing issues"
        if connection_pool._last_error:
            error_message = f"Connection error: {connection_pool._last_error}"
        
        return StatusResponse(
            status="healthy" if all_services_online else "degraded",
            services=status,
            message=error_message
        )
        
    except Exception as e:
        logger.error(f"Error checking service status: {str(e)}")
        return StatusResponse(
            status="error",
            services={
                "gemini": {
                    "status": "unknown",
                    "last_checked": None,
                    "error": str(e) or connection_pool._last_error
                },
                "openai": {
                    "status": "unknown",
                    "last_checked": None,
                    "error": str(e) or connection_pool._last_error,
                    "gpt4_available": False
                }
            },
            message=f"Unable to check service status: {str(e)}"
        )

@app.post("/translate")
async def translate_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model_type = data.get("model", "GPT35")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(text) > 30000:  # Reduced limit for safety
            raise HTTPException(status_code=400, detail="Text too long (max 30000 characters)")
        
        # Map frontend model names to backend model types
        model_mapping = {
            "gemini-1.5-pro": ModelType.GEMINI2.value,
            "gemini-1.5-flash-8b": ModelType.GEMINI.value,
            "gpt-3.5-turbo": ModelType.GPT35.value,
            "gpt-4": ModelType.GPT4.value
        }
        
        model_type = model_mapping.get(model_type, ModelType.GPT35.value)
        
        if model_type in [ModelType.GEMINI.value, ModelType.GEMINI2.value]:
            try:
                # Configure Gemini
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                
                # Try to use the specified model directly
                try:
                    logger.info(f"Attempting to use {model_type} model")
                    model = genai.GenerativeModel(model_type)
                    
                    # Create a proper translation prompt
                    prompt = TRANSLATION_PROMPT.format(text=text)
                    
                    # Generate translation with retry logic
                    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
                    def generate_with_retry():
                        return model.generate_content(prompt)
                    
                    loop = asyncio.get_event_loop()
                    response = await asyncio.wait_for(
                        loop.run_in_executor(None, generate_with_retry),
                        timeout=30.0
                    )
                    
                    if not response or not response.text:
                        raise ValueError("Empty response from Gemini API")
                        
                    logger.info("Successfully generated translation")
                    return {"translated_text": response.text}
                    
                except Exception as e:
                    logger.error(f"Failed to use {model_type} model: {str(e)}")
                    raise
                    
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Gemini API timeout")
            except Exception as e:
                logger.error(f"Gemini translation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            try:
                # Create a proper translation prompt
                prompt = TRANSLATION_PROMPT.format(text=text)
                
                # Generate translation with retry logic
                @backoff.on_exception(backoff.expo, Exception, max_tries=3)
                def generate_with_retry():
                    return openai_client.chat.completions.create(
                        model=model_type,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=4000
                    )
                
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, generate_with_retry),
                    timeout=30.0
                )
                
                if not response or not response.choices:
                    raise ValueError("Empty response from OpenAI API")
                    
                logger.info("Successfully generated translation")
                return {"translated_text": response.choices[0].message.content}
                
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="OpenAI API timeout")
            except Exception as e:
                logger.error(f"OpenAI translation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import socket
    
    PORT = 8090  # Update port to match the one being used
    MAX_PORT_ATTEMPTS = 10
    
    try:
        # Try to find an available port
        for port_attempt in range(PORT, PORT + MAX_PORT_ATTEMPTS):
            try:
                # Test if port is in use
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port_attempt))
                    PORT = port_attempt
                    break
            except OSError as e:
                logger.error(f"Port {port_attempt} is not available: {str(e)}")
                if port_attempt == PORT + MAX_PORT_ATTEMPTS - 1:
                    logger.error(f"Could not find an available port in range {PORT}-{PORT + MAX_PORT_ATTEMPTS}")
                    raise
                continue
        
        logger.info(f"Starting server on port {PORT}")
        
        # Check API connections
        openai_status, gemini_status, gemini2_status = asyncio.run(test_api_connections())
        logger.info("=== API Connection Status ===")
        logger.info(f"OpenAI API: {openai_status}")
        logger.info(f"Gemini 1.5 API: {gemini_status}")
        logger.info(f"Gemini 2.0 API: {gemini2_status}")
        logger.info("==========================")
        
        # Start the server with more detailed logging
        uvicorn.run(
            "translation_bot:app",
            host="0.0.0.0",
            port=PORT,
            reload=True,
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 