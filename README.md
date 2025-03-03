# Persian to English Translator

A web-based Persian to English translation service using OpenAI's GPT model.

## Features

- Real-time translation from Persian to English
- Clean and responsive web interface
- Support for RTL text input
- Error handling and loading states

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd translation_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python translation_bot.py
```

5. Open your browser and visit: `http://localhost:8088`

## Requirements

- Python 3.8+
- FastAPI
- OpenAI API key
- Other dependencies listed in `requirements.txt`

## Usage

1. Enter Persian text in the text area
2. Click "Translate" or press Enter
3. The English translation will appear below

## License

MIT License 