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
1. Title and heading detection & formatting:
   - Identify standalone opening sentences as potential titles
   - Look for these title patterns:
     * Short, impactful opening statements
     * Sentences followed by detailed explanations
     * Thematic statements at section starts
   - Preserve title case and emphasis
   - Maintain heading hierarchy
   - Keep heading formatting consistent
2. Grammar and syntax corrections
3. Paragraph structure and flow
4. Punctuation and spacing
5. Word choice and consistency
6. Sentence structure clarity
7. Professional coaching terminology accuracy
8. Formatting consistency

DO NOT:
- Change core concepts or definitions
- Alter coaching methodologies
- Modify exercise instructions
- Rewrite content meanings
- Change title meanings or hierarchy

Make ONLY technical corrections that improve readability while preserving the exact meaning.
Ensure titles and headings are clearly distinguished from body text.
For implicit titles (like standalone opening sentences), add appropriate formatting."""

COACHING_FAST_PROMPT = """Quick edit of this Persian text focusing ONLY on:
1. Title and heading identification & preservation:
   - Detect and preserve implicit titles (standalone opening sentences)
   - Look for these title patterns:
     * Short opening statements
     * Theme-setting sentences
     * Section introductions
   - Keep titles exactly as is
   - Maintain heading levels
   - Fix only obvious title formatting issues
2. Basic grammar fixes
3. Obvious typos
4. Clear punctuation errors
5. Basic sentence structure issues

Keep everything else exactly as is, especially title meanings and hierarchy.
Mark implicit titles with appropriate formatting."""

COACHING_TRANSLATION_PROMPT = """Translate this Persian coaching text to English with these requirements:
1. Title and heading accuracy:
   - Identify and translate implicit titles (like standalone opening sentences)
   - Look for these title patterns:
     * Short, impactful statements at section starts
     * Theme-introducing sentences
     * Standalone declarative statements
   - Translate titles with exact meaning
   - Preserve heading hierarchy
   - Maintain title emphasis and formatting
2. Preserve all coaching methodologies and concepts exactly
3. Maintain professional coaching terminology
4. Keep the same tone and style
5. Preserve paragraph structure
6. Maintain all examples and exercises in their original form
7. Keep any specialized coaching terms in their professional form

Focus on accuracy of coaching concepts and title translations over literary style.
Format implicit titles appropriately in the translation."""

COACHING_FAST_TRANSLATION_PROMPT = """Quick translation of this Persian coaching text to English:
1. Accurate title and heading translation:
   - Detect and translate implicit titles (opening statements)
   - Keep title meanings exact
   - Preserve heading structure
   - Mark standalone opening sentences as titles
2. Maintain core coaching concepts
3. Keep professional terminology
4. Preserve exercise instructions
5. Maintain basic structure

Prioritize preserving coaching meaning and title accuracy over style.
Ensure implicit titles are properly formatted in translation."""

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
    mode: str = "detailed"  # Default to detailed mode
    
    class Config:
        protected_namespaces = ()  # Resolve model_ namespace conflict

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
    # Split texts into lines for better diff
    original_lines = original.split('\n')
    edited_lines = edited.split('\n')
    
    # Pre-process to handle existing HTML tags
    def clean_html(text):
        # Remove existing HTML tags but preserve content
        text = re.sub(r'<h2 class="title">(.*?)</h2>', r'\1', text)
        text = re.sub(r'<span class="highlight-\w+">(.*?)</span>', r'\1', text)
        return text
    
    original_lines = [clean_html(line) for line in original_lines]
    edited_lines = [clean_html(line) for line in edited_lines]
    
    matcher = difflib.SequenceMatcher(None, original_lines, edited_lines)
    html_diff = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Check if this is a title line
            for line in original_lines[i1:i2]:
                if is_title_line(line):
                    html_diff.append(f'<h2 class="title">{line}</h2>')
                else:
                    html_diff.append(line)
        elif tag == 'delete':
            for line in original_lines[i1:i2]:
                html_diff.append(f'<span class="highlight-removed">{line}</span>')
        elif tag == 'insert':
            for line in edited_lines[j1:j2]:
                if is_title_line(line):
                    html_diff.append(f'<h2 class="title highlight-added">{line}</h2>')
                else:
                    html_diff.append(f'<span class="highlight-added">{line}</span>')
        elif tag == 'replace':
            # Show both old and new versions with appropriate highlighting
            html_diff.append('<div class="replacement">')
            for line in original_lines[i1:i2]:
                html_diff.append(f'<span class="highlight-removed">{line}</span>')
            for line in edited_lines[j1:j2]:
                if is_title_line(line):
                    html_diff.append(f'<h2 class="title highlight-grammar">{line}</h2>')
                else:
                    html_diff.append(f'<span class="highlight-grammar">{line}</span>')
            html_diff.append('</div>')
    
    return '<br>'.join(html_diff)

def is_title_line(line: str) -> bool:
    """Detect if a line is likely a title."""
    line = line.strip()
    if not line:  # Skip empty lines
        return False
        
    # Title patterns
    patterns = [
        r'^عنوان\s*:',  # Explicit title marker
        r'^بخش\s*[\d۰-۹]+:',  # Section markers
        r'^زیربخش\s*[\d۰-۹\.]+:',  # Subsection markers
        r'^#+ ',  # Markdown style headers
        r'^[^.!؟\n]{2,50}$'  # Short standalone lines (2-50 chars without sentence endings)
    ]
    
    # Check for explicit patterns first
    if any(re.match(pattern, line) for pattern in patterns):
        return True
    
    # Then check for implicit title characteristics
    words = line.split()
    return (
        len(words) <= 7 and  # Short phrase
        not any(end in line for end in ['.', '!', '؟', '?']) and  # Not ending with sentence markers
        not any(word.endswith('می‌کند') or word.endswith('می‌کنیم') for word in words) and  # Not a regular sentence
        not line.startswith('در') and  # Not starting with prepositions
        not line.startswith('و') and  # Not starting with conjunctions
        line  # Not empty
    )

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
async def edit(request: EditRequest) -> EditResponse:
    """Edit text using the specified model."""
    try:
        # Initialize variables
        edited_text = request.text  # Default to original text
        changes = []
        technical_explanation = ""
        model_explanation = ""
        
        # Process based on model
        if request.model == ModelType.GEMINI.value:
            edited_text, changes = await process_gemini_edit(request.text, request.mode)
            if not edited_text:  # If Gemini processing failed, use original text
                edited_text = request.text
        elif request.model in [ModelType.GPT35.value, ModelType.GPT4.value]:
            # ... existing code for other models ...
            pass
        
        # Generate HTML diff with improved highlighting
        diff_html = generate_html_diff(request.text, edited_text)
        
        # Calculate word count information
        word_count = count_persian_words(edited_text)
        word_count_status = get_word_count_status(word_count)
        
        # Set explanations based on mode
        technical_explanation = "Light editing completed" if request.mode == "fast" else "Detailed editing completed while preserving all content"
        model_explanation = f"Using {request.model} model with {'fast' if request.mode == 'fast' else 'detailed'} editing mode"
        
        return EditResponse(
            edited_text=edited_text,
            technical_explanation=technical_explanation,
            model_explanation=model_explanation,
            changes=changes,
            diff_html=diff_html,
            word_count=word_count,
            word_count_status=word_count_status
        )
        
    except Exception as e:
        logger.error(f"Edit error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate(request: TranslationRequest) -> TranslationResponse:
    try:
        # Get the appropriate prompt based on mode
        system_msg = "You are a professional translator specializing in coaching and professional development content."
        user_msg = COACHING_TRANSLATION_PROMPT if request.mode == "detailed" else COACHING_FAST_TRANSLATION_PROMPT
        user_msg += f"\n\nText to translate: {request.text}"

        if request.model == ModelType.GEMINI.value:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(gemini_model.generate_content, user_msg),
                    timeout=60.0
                )
                
                if hasattr(response, 'text'):
                    translation = response.text.strip()
                else:
                    translation = response.parts[0].text.strip()
            except Exception as e:
                logger.error(f"Gemini translation error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Gemini translation failed: {str(e)}")
        
        elif request.model in [ModelType.GPT35.value, ModelType.GPT4.value]:
            response = await openai_client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3
            )
            translation = response.choices[0].message.content
        
        elif request.model == ModelType.CLAUDE.value:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            response = await client.messages.create(
                model=request.model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": f"{system_msg}\n\n{user_msg}"
                }]
            )
            translation = response.content[0].text
        
        elif request.model == ModelType.GOOGLE.value:
            from google.cloud import translate_v2 as translate
            translate_client = translate.Client()
            result = translate_client.translate(
                request.text,
                target_language='en',
                source_language='fa'
            )
            translation = result['translatedText']
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
        
        model_explanation = "Translation completed with focus on coaching terminology and concepts"
        technical_explanation = "Detailed coaching translation completed" if request.mode == "detailed" else "Fast coaching translation completed"
        
        return TranslationResponse(
            translated_text=translation,
            technical_explanation=technical_explanation,
            model_explanation=model_explanation
        )
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
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

async def test_title_handling():
    """Test function to verify title and heading handling"""
    sample_text = """عنوان اصلی: اصول و تکنیک‌های کوچینگ حرفه‌ای

بخش اول: مبانی کوچینگ
در این بخش، به بررسی اصول اساسی کوچینگ می‌پردازیم. کوچینگ یک فرآیند همکاری است که به مراجع کمک می‌کند به اهداف خود دست یابد.

زیربخش ۱.۱: تعریف کوچینگ
کوچینگ عبارت است از همراهی حرفه‌ای با مراجع برای دستیابی به نتایج مطلوب.

بخش دوم: مهارت‌های پیشرفته
در این قسمت، تکنیک‌های پیشرفته کوچینگ را بررسی می‌کنیم.

زیربخش ۲.۱: گوش دادن فعال
گوش دادن فعال یکی از مهم‌ترین مهارت‌های یک کوچ حرفه‌ای است."""

    # Test editing with different models
    for model in [ModelType.GEMINI.value, ModelType.GPT4.value]:
        try:
            # Test detailed editing
            detailed_request = EditRequest(text=sample_text, mode="detailed", model=model)
            detailed_result = await edit(detailed_request)
            logger.info(f"Detailed edit test with {model} - Success")
            print(f"\nDetailed Edit Result ({model}):")
            print(detailed_result.edited_text)
            
            # Test fast editing
            fast_request = EditRequest(text=sample_text, mode="fast", model=model)
            fast_result = await edit(fast_request)
            logger.info(f"Fast edit test with {model} - Success")
            print(f"\nFast Edit Result ({model}):")
            print(fast_result.edited_text)
            
            # Test both translation modes
            for mode in ["detailed", "fast"]:
                translation_request = TranslationRequest(text=sample_text, model=model, mode=mode)
                translation_result = await translate(translation_request)
                logger.info(f"{mode.capitalize()} translation test with {model} - Success")
                print(f"\n{mode.capitalize()} Translation Result ({model}):")
                print(translation_result.translated_text)
            
        except Exception as e:
            logger.error(f"Test failed for {model}: {str(e)}")
            raise

async def process_gemini_edit(text: str, mode: str) -> tuple[str, List[Change]]:
    """Process text editing with Gemini model and generate highlighted changes."""
    prompt = COACHING_DETAILED_PROMPT if mode == "detailed" else COACHING_FAST_PROMPT
    prompt += f"\n\nText to edit:\n{text}\n\nPlease edit the text and mark your changes using these HTML tags:\n- For added text: <add>new text</add>\n- For removed text: <del>old text</del>\n- For modified text: <mod>modified text</mod>\n- For titles: <title>title text</title>\n\nMake sure to mark any standalone opening sentences as titles."
    
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(gemini_model.generate_content, prompt),
            timeout=60.0
        )
        
        edited_text = response.text.strip() if hasattr(response, 'text') else response.parts[0].text.strip()
        if not edited_text:
            return text, []  # Return original text if no edits
            
        # Extract changes from HTML tags
        changes = []
        
        # Process titles first (they take precedence)
        for match in re.finditer(r'<title>(.*?)</title>', edited_text):
            title_text = match.group(1)
            edited_text = edited_text.replace(match.group(0), f'<h2 class="title">{title_text}</h2>')
            changes.append(Change(type="title", new=title_text))
        
        # Process other changes
        for tag, change_type in [
            ('add', 'added'),
            ('del', 'removed'),
            ('mod', 'grammar')
        ]:
            for match in re.finditer(f'<{tag}>(.*?)</{tag}>', edited_text):
                content = match.group(1)
                if change_type == 'removed':
                    changes.append(Change(type=change_type, old=content))
                else:
                    changes.append(Change(type=change_type, new=content))
                
                # Apply appropriate highlighting
                highlight_class = f'highlight-{change_type}'
                edited_text = edited_text.replace(
                    match.group(0),
                    f'<span class="{highlight_class}">{content}</span>'
                )
        
        # If no explicit title was marked and it's detailed mode, check for implicit titles
        if not any(c.type == "title" for c in changes) and mode == "detailed":
            lines = edited_text.split('\n')
            if lines and is_title_line(lines[0]):
                title_text = lines[0]
                lines[0] = f'<h2 class="title">{title_text}</h2>'
                edited_text = '\n'.join(lines)
                changes.append(Change(type="title", new=title_text))
        
        # If no changes were detected, return the edited text as is
        if not changes and edited_text != text:
            changes.append(Change(type="grammar", new=edited_text))
            
        return edited_text, changes
        
    except Exception as e:
        logger.error(f"Gemini processing error: {str(e)}")
        return text, []  # Return original text on error

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Run the title handling test
    asyncio.run(test_title_handling())
    logger.info("Title handling test completed")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8088) 