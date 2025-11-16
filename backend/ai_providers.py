#!/usr/bin/env python3
"""
AI Providers Module - Support multiple AI providers
Copied from rag_pdf.py with modifications for FastAPI backend
"""
import os
import logging
from typing import List, Dict, Union, Optional, Generator

# Try to import AI provider libraries
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ollama library not available. Install with: pip install ollama")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Google Generative AI library available")
except ImportError as e:
    GEMINI_AVAILABLE = False
    logging.warning(f"⚠️ Google Generative AI library not available: {e}")

logger = logging.getLogger(__name__)

# AI Provider configurations
AI_PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["gemma3:latest", "llama3.1:latest", "qwen2.5:latest", "mistral:latest", "phi3:latest"],
        "api_key_required": False,
        "base_url": "http://localhost:11434",
        "default_model": "gemma3:latest"
    },
    "minimax": {
        "name": "Minimax",
        "models": ["abab6.5", "abab6.5s", "abab5.5"],
        "api_key_required": True,
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "abab6.5"
    },
    "manus": {
        "name": "Manus",
        "models": ["manus-code", "manus-reasoning", "manus-vision"],
        "api_key_required": True,
        "api_key_env": "MANUS_API_KEY",
        "base_url": "https://api.manus.ai/v1",
        "default_model": "manus-code"
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "api_key_required": True,
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "default_model": "gemini-2.5-pro"
    },
    "zhipu": {
        "name": "Zhipu AI (GLM)",
        "models": ["GLM-4.6", "glm-4.6", "glm-4", "glm-4v", "glm-3-turbo"],
        "api_key_required": True,
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "default_model": "GLM-4.6"
    },
    "chatgpt": {
        "name": "ChatGPT (OpenAI)",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_required": True,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o"
    }
}

# Default AI Provider
DEFAULT_AI_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", "ollama")

# Check library availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info("✅ Google Generative AI library available")
except ImportError as e:
    GEMINI_AVAILABLE = False
    logger.warning(f"⚠️ Google Generative AI library not available: {e}")

def get_ai_provider_config(provider_name: str) -> dict:
    """Get AI provider configuration"""
    return AI_PROVIDERS.get(provider_name, AI_PROVIDERS["ollama"])

def get_available_providers() -> List[str]:
    """Get list of available AI providers"""
    providers = []

    # Always include some demo providers for UI testing
    providers.extend(["openai", "gemini", "claude", "minimax"])

    # Also check for actual available providers
    for key, config in AI_PROVIDERS.items():
        if key == "ollama":
            if OLLAMA_AVAILABLE and key not in providers:
                providers.append(key)
        elif config["api_key_required"]:
            api_key = os.getenv(config["api_key_env"])
            if api_key and api_key.strip():
                # Special check for libraries
                if key == "gemini" and not GEMINI_AVAILABLE:
                    continue
                if key == "chatgpt" and not OPENAI_AVAILABLE:
                    continue
                if key not in providers:
                    providers.append(key)
        else:
            if key not in providers:
                providers.append(key)
    return providers

def get_provider_models(provider_name: str) -> List[str]:
    """Get available models for a provider"""
    config = get_ai_provider_config(provider_name)
    return config.get("models", [])

def call_ai_provider(provider_name: str, model: str, messages: List[Dict], stream: bool = True, **kwargs):
    """
    Unified function to call different AI providers

    Args:
        provider_name: Name of the AI provider (ollama, minimax, manus, gemini, chatgpt)
        model: Model name to use
        messages: List of messages in format [{"role": "user", "content": "..."}]
        stream: Whether to stream the response
        **kwargs: Additional parameters for the specific provider

    Returns:
        Response stream or response object
    """
    config = get_ai_provider_config(provider_name)

    if provider_name == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama library not available. Install with: pip install ollama")

        # Use existing Ollama implementation
        try:
            return ollama.chat(
                model=model,
                messages=messages,
                stream=stream,
                options={
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "num_predict": kwargs.get("num_predict", 1500)
                }
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    elif provider_name == "chatgpt":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")

        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            client = openai.OpenAI(api_key=api_key)

            if stream:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    elif provider_name == "gemini":
        if not GEMINI_AVAILABLE:
            error_msg = "Google Generative AI library not available. Please install with: pip install google-generativeai"
            logger.error(error_msg)
            return ({"message": {"content": error_msg}} for _ in range(1))

        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)

            # Convert messages to Gemini format
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"

            prompt += "Assistant: "

            response = model_obj.generate_content(
                prompt,
                stream=stream,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.3),
                    max_output_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            )

            # Convert Gemini stream to match Ollama/OpenAI format
            def gemini_stream_converter(gemini_response):
                for chunk in gemini_response:
                    if chunk.text:
                        yield {"message": {"content": chunk.text}}

            return gemini_stream_converter(response)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    elif provider_name in ["minimax", "manus", "zhipu"]:
        # Generic OpenAI-compatible API implementation
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=config["base_url"]
            )

            if stream:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
        except Exception as e:
            logger.error(f"{provider_name} API error: {e}")
            raise

    else:
        raise ValueError(f"Unsupported AI provider: {provider_name}")

def stream_response(provider_name: str, model: str, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
    """
    Stream response from AI provider

    Yields:
        str: Response chunks
    """
    try:
        response = call_ai_provider(provider_name, model, messages, stream=True, **kwargs)

        if hasattr(response, '__iter__'):
            # Handle streaming responses
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta'):
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                elif hasattr(chunk, 'message'):
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        yield content
                elif hasattr(chunk, 'text'):
                    # Handle Gemini streaming
                    if chunk.text:
                        yield chunk.text
        else:
            # Handle non-streaming response
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content:
                    yield content
            else:
                yield str(response)

    except Exception as e:
        logger.error(f"Error streaming response from {provider_name}: {e}")
        yield f"Error: {str(e)}"

def get_non_stream_response(provider_name: str, model: str, messages: List[Dict], **kwargs) -> str:
    """
    Get non-streaming response from AI provider

    Returns:
        str: Complete response
    """
    try:
        response = call_ai_provider(provider_name, model, messages, stream=False, **kwargs)

        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)

    except Exception as e:
        logger.error(f"Error getting response from {provider_name}: {e}")
        return f"Error: {str(e)}"