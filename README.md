# Persian Text Editor and Translator

A web application for editing Persian text and translating it to English using various AI models.

## Features

- Persian text editing with AI assistance
- Translation to English using multiple models:
  - GPT-3.5 Turbo
  - GPT-4
  - Claude-3
  - Google Translate

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd translation_project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file:
     - `OPENAI_API_KEY`
     - `ANTHROPIC_API_KEY`
     - `GOOGLE_API_KEY`

5. Run the application:
```bash
python translation_bot.py
```

6. Open your browser and navigate to `http://localhost:8088`

## Usage

1. Enter Persian text in the "Write Persian Text" section
2. Click "Edit Text" to improve the text using AI
3. Click "Finalize Edit" to move the text to the translation section
4. Select a translation model
5. Click "Translate" to get the English translation

## Requirements

- Python 3.8+
- FastAPI
- OpenAI API key
- Anthropic API key (for Claude)
- Google Cloud API key (for Translation API)

## License

MIT License 