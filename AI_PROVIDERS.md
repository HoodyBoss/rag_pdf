# AI Provider Configuration

This document explains how to configure and use different AI providers with the RAG PDF application.

## Supported AI Providers

The application supports the following AI providers:

### 1. Ollama (Local)
- **Description**: Run models locally on your machine
- **Models Available**: `gemma3:latest`, `qwen3:latest`, `llama3.2:latest`
- **API Key Required**: No
- **Setup**: Install Ollama on your local machine and pull the models you want to use

### 2. Minimax
- **Description**: Cloud-based AI service
- **Models Available**: `abab6.5`, `abab6.5s`, `abab5.5`
- **API Key Required**: Yes
- **API Key Environment Variable**: `MINIMAX_API_KEY`
- **Base URL**: `https://api.minimax.chat/v1`

### 3. Manus
- **Description**: AI service with code and reasoning capabilities
- **Models Available**: `manus-code`, `manus-reasoning`, `manus-vision`
- **API Key Required**: Yes
- **API Key Environment Variable**: `MANUS_API_KEY`
- **Base URL**: `https://api.manus.ai/v1`

### 4. Google Gemini
- **Description**: Google's generative AI models
- **Models Available**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.0-pro`
- **API Key Required**: Yes
- **API Key Environment Variable**: `GEMINI_API_KEY`
- **Base URL**: `https://generativelanguage.googleapis.com/v1`

### 5. ChatGPT (OpenAI)
- **Description**: OpenAI's GPT models
- **Models Available**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **API Key Required**: Yes
- **API Key Environment Variable**: `OPENAI_API_KEY`
- **Base URL**: `https://api.openai.com/v1`

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root with your API keys:

```env
# AI Provider Configuration
DEFAULT_AI_PROVIDER=ollama

# AI Provider API Keys
MINIMAX_API_KEY=your_minimax_api_key_here
MANUS_API_KEY=your_manus_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Using the Web Interface

1. **API Key Configuration Tab**:
   - Navigate to the "🔑 API Key Configuration" tab
   - Enter your API keys for each provider you want to use
   - Click the test buttons to verify your API keys work
   - Click "บันทึกการตั้งค่า" to save your configuration

2. **Chat Tab**:
   - Select your preferred AI Provider from the dropdown
   - Choose the model you want to use from the model dropdown
   - The model list updates automatically based on the selected provider

## Installation Requirements

Make sure you have the required Python packages installed:

```bash
pip install openai google-generativeai
```

The application will automatically detect which packages are available and only show providers that have the required libraries installed.

## How It Works

1. **Provider Detection**: The application scans for API keys and library availability
2. **Dynamic Model Loading**: Models are loaded dynamically based on the selected provider
3. **Unified Interface**: All providers use the same interface, so you can switch between them seamlessly
4. **Error Handling**: The application provides clear error messages if API keys are missing or invalid

## Getting API Keys

### Minimax
1. Visit the Minimax platform
2. Create an account and get your API key from the dashboard

### Manus
1. Visit the Manus platform
2. Sign up and obtain your API key

### Google Gemini
1. Go to Google AI Studio
2. Create a new API key or use an existing one

### OpenAI
1. Visit the OpenAI platform
2. Create an account and generate an API key from the API keys section

## Troubleshooting

### Common Issues

1. **Provider not showing in dropdown**:
   - Check that you have the required libraries installed
   - Verify your API key is set correctly in the environment or web interface

2. **API connection fails**:
   - Verify your API key is correct and active
   - Check if you have sufficient credits/quota
   - Ensure your internet connection is stable

3. **Model not working**:
   - Some models may have specific requirements or be unavailable in certain regions
   - Check the provider's documentation for model availability

### Error Messages

- `❌ LLM call failed: API key not found`: You need to set the API key for the selected provider
- `❌ Library not available`: Install the required Python package for that provider
- `❌ API Key ไม่ถูกต้อง`: The API key is invalid or expired

## Security Notes

- Keep your API keys secure and never share them publicly
- Use environment variables or the secure web interface to store keys
- Regularly rotate your API keys for better security
- Monitor your API usage to avoid unexpected charges