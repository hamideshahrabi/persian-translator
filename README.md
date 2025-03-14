# Persian Text Editor and Translator

A web application for editing and translating Persian text using various AI models.

## Features

- Persian text editing with grammar and style improvements
- Translation to English using multiple AI models
- Quality scoring for translations and edits
- Dashboard with usage statistics
- Support for multiple AI models:
  - Gemini Pro
  - GPT-3.5 Turbo
  - GPT-4
  - Claude-3
  - Google Cloud Translation

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   GEMINI_API_KEY=your_gemini_key
   ```
5. Run the application:
   ```bash
   uvicorn translation_bot:app --host 0.0.0.0 --port 8080
   ```

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

## Security Considerations

1. Never commit your `.env` file or expose API keys
2. Consider implementing user authentication for your group
3. Monitor API usage to stay within limits
4. Use HTTPS for secure communication

## Support

For issues or questions, please open an issue in the repository. 