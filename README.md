# PersianAI: Smart Publishing Assistant

An intelligent AI-powered publishing assistant specialized in automating and enhancing the editorial and translation workflow for multi-author professional publications. Built for publishing companies to streamline their content processing pipeline.

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

## Key Features

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

## Features

- Multiple model support (GPT-4, GPT-3.5, Claude, Gemini, Google Translate)
- Specialized in coaching content translation and editing
- Title detection and formatting
- Change highlighting and tracking
- Word count tracking
- Fast and detailed editing modes

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd translation_project
```

2. Create and activate a virtual environment:
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
   - Fill in your API keys in `.env`:
     - `OPENAI_API_KEY`: OpenAI API key for GPT models
     - `ANTHROPIC_API_KEY`: Anthropic API key for Claude
     - `GOOGLE_API_KEY`: Google API key for translation
     - `GEMINI_API_KEY`: Google API key for Gemini
     - Optional: `GOOGLE_APPLICATION_CREDENTIALS` for Google Cloud Translation

5. Run the server:
```bash
python translation_bot.py
```

The server will start at `http://localhost:8088`

## Usage

Visit `http://localhost:8088` in your browser to use the web interface.

### Editing Modes

- **Detailed**: Full editing with comprehensive title detection and formatting
- **Fast**: Quick edits focusing on basic improvements and title preservation

### Models

- GPT-4: High-quality editing and translation
- GPT-3.5: Faster, cost-effective option
- Claude: Alternative high-quality model
- Gemini: Google's latest model
- Google Translate: Basic translation service

## Security Notes

- Never commit your `.env` file or any files containing API keys
- Use environment variables for all sensitive credentials
- Keep your API keys secure and rotate them regularly

## Deployment Options

### Option 1: Deploy on Render (Recommended for small groups)

1. Create an account on [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn translation_bot:app --host 0.0.0.0 --port $PORT`
5. Add your environment variables in Render's dashboard
6. Deploy!

### Option 2: Deploy on Heroku

1. Create an account on [Heroku](https://heroku.com)
2. Install Heroku CLI
3. Create a `Procfile`:
   ```
   web: uvicorn translation_bot:app --host 0.0.0.0 --port $PORT
   ```
4. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```
5. Set environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set ANTHROPIC_API_KEY=your_key
   heroku config:set GOOGLE_API_KEY=your_key
   heroku config:set GEMINI_API_KEY=your_key
   ```

### Option 3: Deploy on DigitalOcean App Platform

1. Create an account on [DigitalOcean](https://digitalocean.com)
2. Create a new App
3. Connect your GitHub repository
4. Configure the app:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `uvicorn translation_bot:app --host 0.0.0.0 --port $PORT`
5. Add your environment variables
6. Deploy!

## Support

For issues or questions, please open an issue in the repository. 