# LLM Configuration Issues and Fixes

## Overview

This document outlines the critical problems found in the LLM calling implementation for both OpenAI and OpenAI-compatible providers with CrewAI, along with the fixes applied.

## Problems Identified

### 1. **LiteLLM Model Format Issues**

**Problem**: The code was incorrectly adding `openai/` prefix to model names for OpenAI-compatible providers.

**Location**: `agents.py` lines 80-82 (original)

**Issue**: 
- Original code: `litellm_model = f"openai/{model}"`
- This resulted in model names like `openai/Qwen/Qwen2.5-Coder-32B-Instruct`
- LiteLLM expects the model name directly for OpenAI-compatible endpoints

**Evidence**: 
```
Error: litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=Qwen/Qwen2.5-Coder-32B-Instruct
```

### 2. **Incorrect Parameter Names**

**Problem**: Using wrong parameter names for LiteLLM configuration.

**Location**: `agents.py` lines 93-123 (original)

**Issues**:
- Using `openai_api_base` instead of `base_url`
- Using `openai_api_key` instead of `api_key`
- Multiple fallback attempts with inconsistent parameter names

### 3. **Test Function Inconsistency**

**Problem**: Test function used direct OpenAI client instead of LiteLLM.

**Location**: `agents.py` lines 19-47 (original)

**Issue**: The test function bypassed LiteLLM entirely, so it didn't test the actual configuration that CrewAI would use.

### 4. **Poor Error Handling and Logging**

**Problem**: Limited logging and error handling made debugging difficult.

**Issues**:
- No structured logging
- Limited error context
- Difficult to trace configuration issues

## Fixes Applied

### Fix 1: Correct LiteLLM Model Format

**Changes Made**:
```python
# Before
litellm_model = f"openai/{model}"

# After  
litellm_model = model
```

**Rationale**: LiteLLM automatically detects OpenAI-compatible endpoints when using the model name directly.

### Fix 2: Correct Parameter Names

**Changes Made**:
```python
# Before
return LLM(
    model=litellm_model,
    openai_api_key=openai_api_key,
    openai_api_base=openai_base_url,
    temperature=0.1,
    max_tokens=4000
)

# After
return LLM(
    model=litellm_model,
    api_key=openai_api_key,
    base_url=openai_base_url,
    temperature=0.1,
    max_tokens=4000
)
```

**Rationale**: CrewAI's LLM class expects standard parameter names that match LiteLLM's interface.

### Fix 3: Updated Test Function

**Changes Made**:
```python
# Before: Used direct OpenAI client
import openai
client = openai.OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url
)

# After: Uses LiteLLM
import litellm
response = litellm.completion(
    model=model,
    messages=[{"role": "user", "content": "Hello, respond with 'OK' if you can hear me."}],
    max_tokens=10,
    temperature=0,
    api_key=openai_api_key,
    base_url=openai_base_url
)
```

**Rationale**: Test function now uses the same configuration path as the actual CrewAI implementation.

### Fix 4: Enhanced Logging and Error Handling

**Changes Made**:
- Added structured logging with different levels (INFO, WARNING, ERROR, DEBUG)
- Added detailed error messages with context
- Added fallback strategies with proper logging
- Added traceback information for debugging

**Example**:
```python
logger.info(f"Configuring OpenAI Compatible LLM: {model} at {openai_base_url}")
logger.warning(f"First attempt with base_url failed: {e1}")
logger.error(f"All attempts failed: {e1}, {e2}, {e3}")
```

### Fix 5: Improved Fallback Strategy

**Changes Made**:
1. **Primary**: Use model name directly with `base_url` parameter
2. **Fallback 1**: Use environment variables only
3. **Fallback 2**: Use `openai/` prefix format as last resort

**Rationale**: Multiple fallback strategies ensure compatibility with different CrewAI and LiteLLM versions.

## Testing

### Test Script

A test script (`test_llm_fix.py`) has been created to verify the fixes:

```bash
python test_llm_fix.py
```

### Manual Testing

To test manually:

1. **Set environment variables**:
   ```bash
   export LITELLM_LOG=DEBUG
   ```

2. **Test connection**:
   ```python
   from agents import test_llm_connection
   success = test_llm_connection("OpenAI Compatible", "Qwen/Qwen2.5-Coder-32B-Instruct", "your-api-key", "https://api.hyperbolic.xyz/v1")
   ```

3. **Test LLM client creation**:
   ```python
   from agents import get_llm_client
   client = get_llm_client("OpenAI Compatible", "Qwen/Qwen2.5-Coder-32B-Instruct", "your-api-key", "https://api.hyperbolic.xyz/v1")
   ```

## Configuration Examples

### OpenAI Compatible (Hyperbolic)

```python
llm_provider = "OpenAI Compatible"
model = "Qwen/Qwen2.5-Coder-32B-Instruct"
api_key = "your-hyperbolic-api-key"
base_url = "https://api.hyperbolic.xyz/v1"
```

### OpenAI Compatible (Together AI)

```python
llm_provider = "OpenAI Compatible"
model = "meta-llama/Llama-2-7b-chat-hf"
api_key = "your-together-api-key"
base_url = "https://api.together.xyz/v1"
```

### OpenAI Compatible (Groq)

```python
llm_provider = "OpenAI Compatible"
model = "llama3-8b-8192"
api_key = "your-groq-api-key"
base_url = "https://api.groq.com/openai/v1"
```

## Troubleshooting

### Common Issues

1. **"LLM Provider NOT provided" Error**
   - **Cause**: Incorrect model format
   - **Solution**: Use model name directly without `openai/` prefix

2. **"Invalid response from LLM call" Error**
   - **Cause**: API key or base URL issues
   - **Solution**: Verify API key and base URL are correct

3. **"404 Not Found" Error**
   - **Cause**: Model name is incorrect or not available
   - **Solution**: Check exact model name (case-sensitive)

### Debug Mode

Enable debug logging to see detailed information:

```bash
export LITELLM_LOG=DEBUG
```

### Environment Variables

Ensure these environment variables are set correctly:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-provider.com/v1"
export LITELLM_LOG="DEBUG"
```

## Dependencies

The fixes require these dependencies:

```
crewai==0.134.0
litellm>=1.0.0
openai>=1.75.0
```

## Version Compatibility

- **CrewAI**: 0.134.0
- **LiteLLM**: >=1.0.0
- **OpenAI**: >=1.75.0

## Future Improvements

1. **Configuration Validation**: Add validation for API keys and base URLs
2. **Provider-Specific Handling**: Add specific handling for different providers
3. **Connection Pooling**: Implement connection pooling for better performance
4. **Retry Logic**: Add exponential backoff for failed requests
5. **Metrics**: Add metrics collection for monitoring LLM performance

## Conclusion

These fixes address the core issues with LLM calling in the CrewAI implementation:

1. ✅ **Correct model format** for LiteLLM
2. ✅ **Proper parameter names** for CrewAI LLM class
3. ✅ **Consistent testing** using LiteLLM
4. ✅ **Enhanced logging** for debugging
5. ✅ **Robust fallback strategies** for compatibility

The implementation should now work correctly with OpenAI-compatible providers like Hyperbolic, Together AI, and Groq. 