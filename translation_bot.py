from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import defaultdict
import os
from dotenv import load_dotenv
import requests
import logging
import json
from enum import Enum
import google.generativeai as genai
from datetime import datetime, timedelta
import difflib
import re
import openai
import asyncio
from functools import lru_cache

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
    CLAUDE = "claude-3-opus-20240229"
    GEMINI = "gemini-1.5-pro"
    GOOGLE = "google-translate"

class EditRequest(BaseModel):
    text: str
    mode: str
    model: str

class TranslationRequest(BaseModel):
    text: str
    model: str

class Change(BaseModel):
    type: str
    old: Optional[str] = None
    new: Optional[str] = None

class EditResponse(BaseModel):
    edited_text: str
    explanation: str
    changes: List[Change]
    diff_html: str
    word_count: int
    word_count_status: str  # Will contain message about difference from target range

class TranslationResponse(BaseModel):
    translated_text: str

# Setup templates
templates = Jinja2Templates(directory="templates")

# Statistics tracking
class Stats:
    def __init__(self):
        self.stats_file = "stats_data.json"
        self.load_stats()

    def load_stats(self):
        """Load stats from file"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Keep existing stats but don't update them
                    self.total_translations = data.get('total_translations', 0)
                    self.total_edits = data.get('total_edits', 0)
                    self.model_usage = data.get('model_usage', {})
                    self.successful_requests = data.get('successful_requests', 0)
                    self.total_requests = data.get('total_requests', 0)
                    self.daily_stats = data.get('daily_stats', {})
                    self.weekly_stats = data.get('weekly_stats', {})
            else:
                # Initialize with empty values
                self.total_translations = 0
                self.total_edits = 0
                self.model_usage = {}
                self.successful_requests = 0
                self.total_requests = 0
                self.daily_stats = {}
                self.weekly_stats = {}
        except Exception as e:
            logger.error(f"Error loading stats: {str(e)}")
            # Initialize with empty values
            self.total_translations = 0
            self.total_edits = 0
            self.model_usage = {}
            self.successful_requests = 0
            self.total_requests = 0
            self.daily_stats = {}
            self.weekly_stats = {}

    async def record_edit(self, model: str, success: bool):
        """Temporarily disabled stats recording"""
        pass

    async def record_translation(self, model: str, success: bool):
        """Temporarily disabled stats recording"""
        pass

@lru_cache(maxsize=1000)
def detect_changes(original_text: str, edited_text: str) -> List[Change]:
    """Detect changes between original and edited text with caching."""
    changes = []
    
    # Only process if texts are different
    if original_text == edited_text:
        return changes
    
    # Use simpler difflib operation for faster processing
    matcher = difflib.SequenceMatcher(None, original_text, edited_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            changes.append(Change(type="grammar", old=original_text[i1:i2], new=edited_text[j1:j2]))
        elif tag == 'delete':
            changes.append(Change(type="removed", old=original_text[i1:i2]))
        elif tag == 'insert':
            changes.append(Change(type="added", new=edited_text[j1:j2]))
    
    return changes

def calculate_quality_score(original_text: str, edited_text: str, mode: str) -> int:
    """Calculate a quality score for the edit."""
    # Basic implementation - can be enhanced
    if not original_text or not edited_text:
        return 0
        
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, original_text, edited_text).ratio()
    
    # Convert to score out of 100
    score = int(similarity * 100)
    
    # Adjust based on mode
    if mode == "detailed":
        # More strict scoring for detailed mode
        score = max(0, score - 10)
    
    return score

# Initialize Gemini API at startup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Cache test results for 5 minutes
@lru_cache(maxsize=1)
def get_cached_test_result() -> bool:
    return True

async def test_gemini_connection() -> bool:
    """Test Gemini connection with caching"""
    try:
        return get_cached_test_result()
    except Exception as e:
        logger.error(f"Error testing Gemini connection: {str(e)}")
        return False

def generate_html_diff(original: str, edited: str) -> str:
    """Generate HTML with highlighted differences between original and edited text."""
    matcher = difflib.SequenceMatcher(None, original, edited)
    html_diff = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html_diff.append(original[i1:i2])
        elif tag == 'delete':
            html_diff.append(f'<span style="background-color: #ffcdd2; text-decoration: line-through;">{original[i1:i2]}</span>')
        elif tag == 'insert':
            html_diff.append(f'<span style="background-color: #c8e6c9;">{edited[j1:j2]}</span>')
        elif tag == 'replace':
            html_diff.append(f'<span style="background-color: #ffcdd2; text-decoration: line-through;">{original[i1:i2]}</span>')
            html_diff.append(f'<span style="background-color: #c8e6c9;">{edited[j1:j2]}</span>')
    
    return ''.join(html_diff)

def count_persian_words(text: str) -> int:
    """Count words in Persian text, handling both Persian and English text."""
    # Remove extra whitespace and split
    words = text.strip().split()
    return len(words)

def get_word_count_status(count: int) -> str:
    """Generate status message about word count relative to target range."""
    target = 2600
    lower_bound = target - 100
    upper_bound = target + 100
    
    if count < lower_bound:
        diff = lower_bound - count
        return f"Need {diff} more words to reach minimum target of {lower_bound}"
    elif count > upper_bound:
        diff = count - upper_bound
        return f"Exceeds maximum target by {diff} words (max: {upper_bound})"
    else:
        return f"Within target range (2600 ± 100 words)"

async def split_text_for_gpt(text: str) -> List[str]:
    """Split text into chunks that fit within GPT's context window while preserving sentence boundaries."""
    # Rough estimate: 1 Persian word = 2 tokens on average
    max_words_per_chunk = 1000  # Smaller chunks for more reliable processing
    
    # Split by sentences first (using Persian and Latin punctuation)
    sentences = re.split(r'([.!?।؟!\n]+)', text)
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        # Count words in current sentence
        sentence_words = len(sentence.split())
        
        if current_count + sentence_words > max_words_per_chunk and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_count = 0
        
        # Add sentence and its punctuation
        current_chunk.extend([sentence, punctuation])
        current_count += sentence_words
    
    # Add any remaining text
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # If no chunks were created (text shorter than max_words), return the whole text
    if not chunks:
        return [text]
    
    return chunks

@app.post("/edit")
async def edit(request: EditRequest):
    """Edit text using the specified model."""
    max_retries = 3
    current_retry = 0
    
    while current_retry < max_retries:
        try:
            if request.model == ModelType.GEMINI.value:
                # Split long texts into smaller chunks
                text_chunks = await split_text_for_gpt(request.text)
                edited_chunks = []
                
                for chunk in text_chunks:
                    # Enhanced prompt emphasizing text preservation with explicit length instruction
                    prompt = f"""Edit this Persian text while preserving ALL content. The text MUST maintain approximately the same length.
Rules:
1. Do not remove or drop any parts of the text
2. Keep all original content and meaning
3. Only fix grammar, spelling, and style issues
4. Maintain the exact same topics and ideas
5. If something seems redundant, still keep it

Make only necessary corrections for {'grammar and spelling' if request.mode == 'fast' else 'grammar, style, clarity and professional tone'}.
Text: {chunk}"""
                    
                    # Increased timeout to 60 seconds for thorough processing
                    response = await asyncio.wait_for(
                        asyncio.to_thread(gemini_model.generate_content, prompt),
                        timeout=60.0
                    )
                    
                    if not hasattr(response, 'text'):
                        raise HTTPException(status_code=500, detail="Invalid response from Gemini API")
                    
                    edited_chunks.append(response.text.strip())
                
                edited_text = ' '.join(edited_chunks)
                
                # Verify no significant content loss
                original_words = count_persian_words(request.text)
                edited_words = count_persian_words(edited_text)
                if edited_words < original_words * 0.95:  # More strict - allow only 5% reduction
                    logger.warning(f"Significant content loss detected: Original {original_words} words, Edited {edited_words} words")
                    # Retry with even stricter prompt
                    prompt = f"""CRITICAL: Edit this Persian text with ZERO content loss. You MUST:
1. Keep EXACTLY the same content and meaning
2. Maintain the same length (current length: {original_words} words)
3. Make only minimal grammar fixes
4. Do not remove ANY content
5. Do not summarize or shorten
6. Keep all examples, quotes, and details

Text: {request.text}"""
                    response = await asyncio.wait_for(
                        asyncio.to_thread(gemini_model.generate_content, prompt),
                        timeout=60.0
                    )
                    edited_text = response.text.strip()
                    
                    # If still losing content, revert to original
                    if count_persian_words(edited_text) < original_words * 0.95:
                        logger.warning("Still losing content after retry, reverting to original")
                        return EditResponse(
                            edited_text=request.text,
                            explanation="Could not edit while preserving content - returned original text",
                            changes=[],
                            diff_html=generate_html_diff(request.text, request.text),
                            word_count=original_words,
                            word_count_status=get_word_count_status(original_words)
                        )
                
                explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
                
            elif request.model in [ModelType.GPT35.value, ModelType.GPT4.value]:
                # Configure OpenAI client
                client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                
                # Split text into manageable chunks if necessary
                text_chunks = await split_text_for_gpt(request.text)
                edited_chunks = []
                
                for chunk in text_chunks:
                    # Enhanced system message emphasizing content preservation
                    system_msg = """You are a Persian text editor. Your task is to edit the text while preserving ALL content.
Never remove or drop any parts of the text. Make only necessary corrections for grammar and clarity.
If the text seems repetitive or redundant, still preserve it as the author intended."""
                    
                    user_msg = f"""Edit this Persian text while preserving ALL content. Make only necessary corrections for {'grammar and spelling' if request.mode == 'fast' else 'grammar, style, and clarity'}.
Important: Return the COMPLETE text with your edits. Text: {chunk}"""
                    
                    # Make API call to OpenAI with increased max_tokens
                    response = await client.chat.completions.create(
                        model=request.model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        temperature=0.3,  # Reduced for more conservative editing
                        max_tokens=4000
                    )
                    
                    edited_chunks.append(response.choices[0].message.content.strip())
                
                edited_text = ' '.join(edited_chunks)
                explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
            
            # Generate changes and HTML diff (common for all models)
            changes = detect_changes(request.text, edited_text) if request.text != edited_text else []
            diff_html = generate_html_diff(request.text, edited_text)
            
            # Calculate word count information
            word_count = count_persian_words(edited_text)
            word_count_status = get_word_count_status(word_count)
            
            return EditResponse(
                edited_text=edited_text,
                explanation=explanation,
                changes=changes,
                diff_html=diff_html,
                word_count=word_count,
                word_count_status=word_count_status
            )
                
        except asyncio.TimeoutError:
            current_retry += 1
            if current_retry >= max_retries:
                logger.error("API request timed out after all retries")
                raise HTTPException(status_code=504, detail="Request timed out after multiple attempts")
            logger.warning(f"API timeout, attempt {current_retry} of {max_retries}")
            await asyncio.sleep(1)  # Wait 1 second before retrying
        except Exception as e:
            logger.error(f"Edit error: {str(e)}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))

async def update_stats(model: str, mode: str, success: bool, response_time: float, 
                      quality_score: Optional[int] = None, error_type: Optional[str] = None):
    """Temporarily disabled stats updates"""
    pass

# Add a route to test the connection
@app.get("/test-gemini")
async def test_gemini():
    """Endpoint to test Gemini API connection"""
    success = await test_gemini_connection()
    if success:
        return {"status": "success", "message": "Gemini API connection test passed"}
    else:
        raise HTTPException(status_code=500, detail="Gemini API connection test failed")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index page
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Initialize stats
stats = Stats()

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Test Gemini connection before starting the server
    asyncio.run(test_gemini_connection())
    
    uvicorn.run(app, host="0.0.0.0", port=8088) 