# Model Limitations

This document outlines the limitations for each model used in the translation and editing system.

## OpenAI Models

### GPT-3.5-turbo
- Maximum tokens: 30,000
- Maximum chunk size: 1,500 characters
- Maximum text length: 15,000 characters (approximately 3000 words)
- Maximum output tokens: 4,000
- Best for: Fast and reliable processing with good accuracy
- Estimated processing time for 3000 words: 2-3 minutes

### GPT-4
- Maximum tokens: 50,000
- Maximum chunk size: 2,500 characters
- Maximum text length: 21,000 characters (approximately 3000 words)
- Maximum output tokens: 4,000
- Best for: Most accurate processing, better understanding of context and nuances
- Estimated processing time for 3000 words: 1.5-2 minutes

## Gemini Models

### Gemini 1.5 Flash (gemini-1.5-flash-8b)
- Maximum tokens: 1,000,000
- Maximum output tokens: 1,024
- Maximum text length: 15,000 characters (approximately 3000 words)
- Best for: Fast and efficient processing with good accuracy
- Estimated processing time for 3000 words: 1-1.5 minutes

### Gemini 1.5 Pro (gemini-1.5-pro-latest)
- Maximum tokens: 2,000,000
- Maximum output tokens: 1,024
- Maximum text length: 21,000 characters (approximately 3000 words)
- Best for: Advanced AI model with strong understanding of Persian language
- Estimated processing time for 3000 words: 1-1.5 minutes

## Notes
- All models have built-in chunking mechanisms to handle long texts
- Word count validation ensures edited text stays within 20 words of the original
- Timeout limits are set to 60 seconds for translation requests
- Each model has retry mechanisms for failed requests
- For texts longer than 1000 words, Gemini models are recommended for better performance
- Processing times are estimates and may vary based on text complexity and API response times 