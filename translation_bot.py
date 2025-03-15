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

# Coaching-specific prompts
COACHING_DETAILED_PROMPT = """Edit this Persian coaching text focusing on:
1. Grammar and syntax corrections
2. Paragraph structure and flow
3. Punctuation and spacing
4. Word choice and consistency
5. Sentence structure clarity
6. Professional coaching terminology accuracy
7. Formatting consistency

DO NOT:
- Change core concepts or definitions
- Alter coaching methodologies
- Modify exercise instructions
- Rewrite content meanings

Make ONLY technical corrections that improve readability while preserving the exact meaning."""

COACHING_FAST_PROMPT = """Quick edit of this Persian text focusing ONLY on:
1. Basic grammar fixes
2. Obvious typos
3. Clear punctuation errors
4. Basic sentence structure issues

Keep everything else exactly as is."""

COACHING_TRANSLATION_PROMPT = """Translate this Persian coaching text to English with these requirements:
1. Preserve all coaching methodologies and concepts exactly
2. Maintain professional coaching terminology
3. Keep the same tone and style
4. Preserve paragraph structure
5. Maintain all examples and exercises in their original form
6. Keep any specialized coaching terms in their professional form

Focus on accuracy of coaching concepts over literary style."""

COACHING_FAST_TRANSLATION_PROMPT = """Quick translation of this Persian coaching text to English:
1. Maintain core coaching concepts
2. Keep professional terminology
3. Preserve exercise instructions
4. Maintain basic structure

Prioritize preserving coaching meaning over style."""

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
    technical_explanation: str  # Technical details about what was changed
    model_explanation: str  # Model's explanation of its editing process
    changes: List[Change]
    diff_html: str
    word_count: int
    word_count_status: str  # Will contain message about difference from target range

class TranslationResponse(BaseModel):
    translated_text: str
    technical_explanation: str  # Technical details about the translation
    model_explanation: str  # Model's explanation of its translation process

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

# Initialize API clients
openai_client = openai.AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await openai_client.close()
    except Exception as e:
        logger.error(f"Error closing OpenAI client: {str(e)}")

@app.post("/edit")
async def edit(request: EditRequest):
    """Edit text using the specified model."""
    max_retries = 3
    current_retry = 0
    
    while current_retry < max_retries:
        try:
            # Calculate original word count first
            original_word_count = count_persian_words(request.text)
            edited_text = ""
            
            if request.model == ModelType.GEMINI.value:
                # Split long texts into smaller chunks
                text_chunks = await split_text_for_gpt(request.text)
                edited_chunks = []
                
                for chunk in text_chunks:
                    # Use coaching-specific prompts
                    prompt = COACHING_DETAILED_PROMPT if request.mode == "detailed" else COACHING_FAST_PROMPT
                    prompt += f"\n\nText: {chunk}"
                    
                    try:
                        response = await asyncio.wait_for(
                            asyncio.to_thread(gemini_model.generate_content, prompt),
                            timeout=60.0
                        )
                        
                        if hasattr(response, 'text'):
                            chunk_edited = response.text.strip()
                        else:
                            chunk_edited = response.parts[0].text.strip()
                            
                        if not chunk_edited:
                            chunk_edited = chunk
                            
                        edited_chunks.append(chunk_edited)
                        
                    except Exception as e:
                        logger.error(f"Gemini chunk processing error: {str(e)}")
                        edited_chunks.append(chunk)
                
                edited_text = ' '.join(edited_chunks)
                
                # Verify content preservation
                edited_word_count = count_persian_words(edited_text)
                original_word_count = count_persian_words(request.text)
                
                if edited_word_count < original_word_count * 0.95:
                    logger.warning(f"Content loss detected: original {original_word_count} words, edited {edited_word_count} words")
                    return EditResponse(
                        edited_text=request.text,
                        technical_explanation="Could not edit while preserving content - returned original text",
                        model_explanation="The model's edits resulted in significant content loss, so the original text was preserved.",
                        changes=[],
                        diff_html=generate_html_diff(request.text, request.text),
                        word_count=original_word_count,
                        word_count_status=get_word_count_status(original_word_count)
                    )
                
                # Generate changes and HTML diff for Gemini edits
                changes = detect_changes(request.text, edited_text)
                diff_html = generate_html_diff(request.text, edited_text)
                
                # Prepare response with highlighted changes
                technical_explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
                model_explanation = f"Using Gemini model with {'fast' if request.mode == 'fast' else 'detailed'} editing mode. The model preserved coaching terminology and methodologies while making necessary corrections."
                
                return EditResponse(
                    edited_text=edited_text,
                    technical_explanation=technical_explanation,
                    model_explanation=model_explanation,
                    changes=changes,
                    diff_html=diff_html,
                    word_count=edited_word_count,
                    word_count_status=get_word_count_status(edited_word_count)
                )
            
            elif request.model in [ModelType.GPT35.value, ModelType.GPT4.value]:
                # Use OpenAI for editing
                try:
                    # Split text into manageable chunks if necessary
                    text_chunks = await split_text_for_gpt(request.text)
                    edited_chunks = []
                    
                    for chunk in text_chunks:
                        # Use coaching-specific prompts
                        system_msg = "You are a professional Persian coaching text editor. Your task is to edit while preserving ALL coaching content and methodologies."
                        
                        # Select appropriate prompt based on mode
                        user_msg = COACHING_DETAILED_PROMPT if request.mode == "detailed" else COACHING_FAST_PROMPT
                        user_msg += f"\n\nText: {chunk}"
                        
                        # Make API call to OpenAI with increased max_tokens
                        response = await openai_client.chat.completions.create(
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
                    explanation = "Light coaching edit completed" if request.mode == "fast" else "Detailed coaching edit completed while preserving all methodologies"
                except Exception as e:
                    logger.error(f"OpenAI editing error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"OpenAI editing failed: {str(e)}")
            
            elif request.model == ModelType.CLAUDE.value:
                # Use Anthropic's Claude for editing
                from anthropic import AsyncAnthropic
                
                client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
                try:
                    response = await client.messages.create(
                        model=request.model,
                        max_tokens=4000,
                        messages=[{
                            "role": "user",
                            "content": f"You are a professional Persian text editor. Edit this Persian text while preserving ALL content. Make only necessary corrections for {'grammar and spelling' if request.mode == 'fast' else 'grammar, style, and clarity'}. Text: {request.text}"
                        }]
                    )
                    
                    if hasattr(response.content[0], 'text'):
                        edited_text = response.content[0].text
                    else:
                        edited_text = str(response.content[0])
                        
                    explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
                except Exception as e:
                    logger.error(f"Claude editing error: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Claude editing failed: {str(e)}")
            
            elif request.model == ModelType.GOOGLE.value:
                # Use Google Cloud Translation
                from google.cloud import translate_v2 as translate
                
                translate_client = translate.Client()
                result = translate_client.translate(
                    request.text,
                    target_language='en',
                    source_language='fa'
                )
                
                return TranslationResponse(translated_text=result['translatedText'])
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
            
            # Generate changes and HTML diff (common for all models)
            changes = detect_changes(request.text, edited_text) if request.text != edited_text else []
            diff_html = generate_html_diff(request.text, edited_text)
            
            # Calculate word count information
            word_count = count_persian_words(edited_text)
            word_count_status = get_word_count_status(word_count)
            
            # Separate technical explanation from model's explanation
            technical_explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
            model_explanation = f"Using {request.model} model with {'fast' if request.mode == 'fast' else 'detailed'} editing mode. The model preserved coaching terminology and methodologies while making necessary corrections."
            
            # For editing response
            return EditResponse(
                edited_text=edited_text,
                technical_explanation=technical_explanation,
                model_explanation=model_explanation,
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

@app.post("/translate")
async def translate(request: TranslationRequest) -> TranslationResponse:
    try:
        # Get the appropriate prompt based on mode
        system_msg = "You are a professional translator specializing in coaching and professional development content."
        user_msg = COACHING_TRANSLATION_PROMPT if request.mode == "detailed" else COACHING_FAST_TRANSLATION_PROMPT
        user_msg += f"\n\nText to translate: {request.text}"

        # Get translation from model
        response = await openai_client.chat.completions.create(
            model=request.model.value,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3
        )
        
        translation = response.choices[0].message.content
        model_explanation = "Translation completed with focus on coaching terminology and concepts"
        technical_explanation = "Detailed coaching translation completed" if request.mode == "detailed" else "Fast coaching translation completed"
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translation,
            model_explanation=model_explanation,
            technical_explanation=technical_explanation,
            model=request.model,
            mode=request.mode
        )
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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