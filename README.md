# PersianAI: Smart Publishing Assistant

A professional-grade Persian text editor and translator specialized in coaching and psychology content. Features advanced AI-powered editing, translation, and content optimization with support for multiple language models (GPT-4, GPT-3.5, Gemini Pro).

## Overview

PersianAI is a sophisticated publishing automation tool that learns from professional editors and translators to deliver consistent, high-quality results. It significantly reduces processing time while maintaining or improving upon human-level quality standards.

### Key Benefits

- **Time & Resource Optimization**
  - Reduces editorial processing time by up to 80%
  - Automates repetitive tasks while maintaining quality
  - Enables faster publication cycles
  - Frees up editorial team for higher-value tasks

- **Smart Learning System**
  - Continuously learns from editorial decisions
  - Adapts to company's style guidelines
  - Improves with each use
  - Maintains consistency across multiple authors

- **Professional Quality Assurance**
  - Ensures consistent terminology across publications
  - Maintains professional tone and style
  - Preserves author voice while improving readability
  - Validates content against publishing standards

## Features

- **AI-Powered Editorial Processing**
  - Smart content optimization
  - Professional coaching and psychology content enhancement
  - Grammar and style refinement
  - Title and structure preservation
  - Word count validation

- **Intelligent Translation System**
  - High-quality Persian to English translation
  - Context-aware terminology handling
  - Multi-author consistency maintenance
  - Professional publishing standards compliance

- **Advanced Automation Features**
  - Real-time change tracking and highlighting
  - Fast and detailed editing modes
  - Multiple AI model support (GPT-4, GPT-3.5, Gemini Pro)
  - Batch processing capabilities

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
     - `GEMINI_API_KEY`

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
- Gemini API key

## License

MIT License 