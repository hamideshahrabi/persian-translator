# Changes to apply to the working version

# 1. Import our improved prompts
from improvements import TRANSLATION_PROMPT, EDIT_PROMPT, EDIT_PROMPT_DETAILED

# 2. Update the process_gemini_edit function
async def process_gemini_edit(text: str) -> str:
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Try to use the specified model directly
        try:
            logger.info("Attempting to use gemini-1.5-flash-8b model")
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            
            # Create a proper translation prompt
            prompt = EDIT_PROMPT.format(text=text)
            
            # Generate translation with retry logic
            @backoff.on_exception(backoff.expo, Exception, max_tries=3)
            def generate_with_retry():
                return model.generate_content(prompt)
            
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, generate_with_retry),
                timeout=30.0
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")
                
            logger.info("Successfully generated translation")
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to use gemini-1.5-flash-8b model: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Gemini translation error: {str(e)}")
        raise

# 3. Update the process_openai_edit function
async def process_openai_edit(text: str, model: str = ModelType.GPT35.value) -> str:
    try:
        # Create a proper translation prompt
        prompt = EDIT_PROMPT.format(text=text)
        
        # Generate translation with retry logic
        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        def generate_with_retry():
            return openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
        
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, generate_with_retry),
            timeout=30.0
        )
        
        if not response or not response.choices:
            raise ValueError("Empty response from OpenAI API")
            
        logger.info("Successfully generated translation")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {str(e)}")
        raise

# 4. Update the translate_text endpoint
@app.post("/translate")
async def translate_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model_type = data.get("model", "GPT35")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(text) > 30000:  # Reduced limit for safety
            raise HTTPException(status_code=400, detail="Text too long (max 30000 characters)")
        
        # Map frontend model names to backend model types
        model_mapping = {
            "gemini-1.5-pro": ModelType.GEMINI2.value,
            "gemini-1.5-flash-8b": ModelType.GEMINI.value,
            "gpt-3.5-turbo": ModelType.GPT35.value,
            "gpt-4": ModelType.GPT4.value
        }
        
        model_type = model_mapping.get(model_type, ModelType.GPT35.value)
        
        if model_type in [ModelType.GEMINI.value, ModelType.GEMINI2.value]:
            try:
                # Configure Gemini
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                
                # Try to use the specified model directly
                try:
                    logger.info(f"Attempting to use {model_type} model")
                    model = genai.GenerativeModel(model_type)
                    
                    # Create a proper translation prompt
                    prompt = TRANSLATION_PROMPT.format(text=text)
                    
                    # Generate translation with retry logic
                    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
                    def generate_with_retry():
                        return model.generate_content(prompt)
                    
                    loop = asyncio.get_event_loop()
                    response = await asyncio.wait_for(
                        loop.run_in_executor(None, generate_with_retry),
                        timeout=30.0
                    )
                    
                    if not response or not response.text:
                        raise ValueError("Empty response from Gemini API")
                        
                    logger.info("Successfully generated translation")
                    return {"translated_text": response.text}
                    
                except Exception as e:
                    logger.error(f"Failed to use {model_type} model: {str(e)}")
                    raise
                    
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Gemini API timeout")
            except Exception as e:
                logger.error(f"Gemini translation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            try:
                # Create a proper translation prompt
                prompt = TRANSLATION_PROMPT.format(text=text)
                
                # Generate translation with retry logic
                @backoff.on_exception(backoff.expo, Exception, max_tries=3)
                def generate_with_retry():
                    return openai_client.chat.completions.create(
                        model=model_type,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=4000
                    )
                
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, generate_with_retry),
                    timeout=30.0
                )
                
                if not response or not response.choices:
                    raise ValueError("Empty response from OpenAI API")
                    
                logger.info("Successfully generated translation")
                return {"translated_text": response.choices[0].message.content}
                
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="OpenAI API timeout")
            except Exception as e:
                logger.error(f"OpenAI translation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 