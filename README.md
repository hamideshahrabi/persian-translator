README.md – PersianAI Translator
# 🇮🇷 PersianAI Translator
**PersianAI** is an AI-powered editing and translation system built to automate the process of **cleaning and translating Persian (Farsi) text into English** with professional accuracy. Originally developed for a **multi-author coaching book**, this tool ensures consistent tone, grammar, and structure in Persian content before translating it to English using cutting-edge large language models (LLMs).
This project is not just a translation tool — it's a **complete AI editing pipeline** tailored to the unique characteristics of Persian, including formal/informal tone, punctuation, spacing, and linguistic structure.
---
## 🔧 What I Built (As the Developer)
This system was designed and implemented by **Hamideh Shahrabi**, with a focus on solving real-world NLP and translation challenges in publishing.
- ✅ Designed the end-to-end architecture based on FastAPI, WebSocket, and microservices
- ✅ Built AI-powered editing logic for **Persian grammar correction, tone normalization, and style improvements**
- ✅ Integrated **OpenAI (GPT-3.5/4)** and **Google Gemini (Pro/Flash)** models for translation and editing
- ✅ Engineered **prompt templates** for different content types (literary, technical, coaching tone)
- ✅ Developed a **real-time pipeline** with chunking, caching, and streaming
- ✅ Implemented a **change tracking system** with word-level diffs and HTML visualization
- ✅ Conducted extensive testing, validation, and user acceptance review
- ✅ Designed the system to be scalable and API-friendly for future integrations
---
## 🧩 Key Features
- ✏️ **AI Editing for Persian Text**
  - Fixes grammar, tone, spacing, and punctuation issues
  - Uses Persian-specific rules and style correction techniques
  - Real-time word counter (2400-2600 range)
  - Visual diff highlighting for edited text
  - Word-level edits in combined view
- 🌍 **Context-Aware Translation**
  - Translates Persian to English using OpenAI's GPT and Google's Gemini models
  - Preserves meaning, formality, cultural references, and technical terms
  - Optimized chunk sizes and timeouts for better performance
  - Enhanced content preservation and reliability
- 🔁 **Change Tracking & Version Control**
  - Detects and highlights word-level edits
  - Provides HTML-based visual diffs and rollback support
  - Structured change summaries
  - Enhanced explanation generation in Persian
- 🔄 **Real-Time Translation**
  - WebSocket integration allows live updates on translation progress
  - Designed for large documents with chunk-based processing
  - Improved error handling and logging
  - Performance metrics and monitoring
- 🧠 **Custom Prompt Templates**
  - Different prompt styles for literary, technical, and casual content
  - Supports prompt engineering for specialized use cases
  - Optimized translation and editing temperatures
  - Enhanced model performance characteristics
- ⚙️ **Production-Ready API**
  - FastAPI microservices backend with REST and WebSocket endpoints
  - Built-in rate limiting, input validation, and API key management
  - Enhanced security measures and configuration management
  - Comprehensive error handling and retry mechanisms
---
## 🤖 Model Capabilities
### OpenAI Models
#### GPT-3.5-turbo
- Maximum tokens: 30,000
- Maximum text length: 15,000 characters (~3000 words)
- Maximum output tokens: 4,000
- Best for: Fast and reliable processing
- Processing time: 2-3 minutes for 3000 words
#### GPT-4
- Maximum tokens: 50,000
- Maximum text length: 21,000 characters (~3000 words)
- Maximum output tokens: 4,000
- Best for: Most accurate processing
- Processing time: 1.5-2 minutes for 3000 words
### Gemini Models
#### Gemini 1.5 Flash
- Maximum tokens: 1,000,000
- Maximum text length: 15,000 characters (~3000 words)
- Best for: Fast and efficient processing
- Processing time: 1-1.5 minutes for 3000 words
#### Gemini 1.5 Pro
- Maximum tokens: 2,000,000
- Maximum text length: 21,000 characters (~3000 words)
- Best for: Advanced Persian language understanding
- Processing time: 1-1.5 minutes for 3000 words
---
📂 How to Use This Project
### 1. Clone the Repository
```bash
git clone https://github.com/hamideshahrabi/persian-translator.git
cd persian-translator
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Setup Environment Variables
```bash
cp .env.example .env
# Add your OpenAI and Gemini API keys inside the .env file
```
### 4. Run the FastAPI Server
```bash
uvicorn translation_bot:app --reload
```
This will start the backend at `http://localhost:8000`.
---
### 5. Example API Request
To test a translation:
```bash
curl -X POST "http://localhost:8000/" -H "Content-Type: application/json" -d '{"text": "سلام دوست من", "model_type": "gpt-4"}'
```
To test text editing:
```bash
curl -X POST "http://localhost:8000/edit" -H "Content-Type: application/json" -d '{"text": "سلام دوست من", "model_type": "gemini", "edit_mode": "detailed"}'
```
---
### 6. WebSocket for Real-Time Translation
Use a WebSocket client like `websocat` or Postman to connect to:
```
ws://localhost:8000/ws
```
This allows you to receive real-time updates during long translation tasks.
---
### Output
- The edited Persian text and translated English output will be returned in the response.
- HTML diffs are available for tracking word-level changes.
---
📁 File Purposes & LLM Model Overview
### 🔹 Core Files Overview
| File Name            | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `translation_bot.py` | Main FastAPI application with REST & WebSocket endpoints, CORS configuration, request handling, and real-time translation pipeline |
| `improvements.py`    | Persian text enhancement with grammar correction, punctuation fixing, tone normalization, and style improvements using custom rules and LLM prompts |
| `changes.py`         | Change tracking system with word-level diffs, HTML visualization, and version history support |
| `test_changes.py`    | Comprehensive testing for change tracking accuracy (insert, delete, replace operations) |
| `test_connections.py`| API connection verification for OpenAI and Gemini models with error handling and validation |
| `test_gemini.py`     | Gemini API response validation and test cases for various scenarios |
| `.env.example`       | Environment variable template for API keys and system configuration |
| `requirements.txt`   | Project dependencies including FastAPI, OpenAI, Google Generative AI |
| `model_limitations.md`| Detailed documentation of model capabilities, limitations, and performance |
| `PROJECT_PROGRESS.md`| Complete development timeline and feature implementation history |

---
### 🚀 Recent Updates
- Implemented word-level edits in combined view
- Enhanced paragraph spacing and structure
- Improved title detection in long texts
- Optimized chunk sizes and timeouts
- Enhanced error handling and logging
- Added comprehensive model performance documentation
- Improved content preservation in long texts
- Enhanced security measures and configuration

### 🔄 Known Limitations
1. Model may struggle with maintaining paragraph structure in very long texts
2. Title detection needs improvement in complex documents
3. Word count restrictions need optimization for specific use cases
4. Some content dropping may occur in very long texts (>3000 words)

### 🎯 Next Steps
1. Further improve paragraph spacing and structure
2. Enhance title detection accuracy
3. Optimize model performance for long texts
4. Address word count restrictions
5. Implement additional content preservation measures

---
### 📊 Performance Notes
- All models include built-in chunking for long texts
- Word count validation ensures edited text stays within 20 words of original
- 60-second timeout limits for translation requests
- Automatic retry mechanisms for failed requests
- Gemini models recommended for texts >1000 words
- Processing times vary based on text complexity

For more detailed information about model capabilities and limitations, please refer to `model_limitations.md`.

---
### 🔐 Security Features
- Environment-based API key management
- Secure key rotation system
- Rate limiting per API key
- Input validation and sanitization
- CORS configuration
- Error logging and monitoring

---
### 📚 Documentation
For more detailed information, refer to:
- `PROJECT_PROGRESS.md` - Complete development timeline
- `model_limitations.md` - Detailed model capabilities
- API documentation at `/docs` endpoint
- WebSocket documentation at `/ws/docs`
