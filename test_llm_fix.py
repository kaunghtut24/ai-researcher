#!/usr/bin/env python3
"""
Test script to verify LLM configuration fixes
"""

import os
import sys
from agents import test_llm_connection, get_llm_client

def test_openai_compatible_config():
    """Test OpenAI Compatible configuration"""
    print("Testing OpenAI Compatible Configuration...")
    
    # Test parameters (replace with your actual values)
    llm_provider = "OpenAI Compatible"
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    api_key = "your-api-key-here"
    base_url = "https://api.hyperbolic.xyz/v1"
    
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    
    # Test connection
    print("\n1. Testing LLM connection...")
    success = test_llm_connection(llm_provider, model, api_key, base_url)
    print(f"Connection test result: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print("\n2. Testing LLM client creation...")
        try:
            client = get_llm_client(llm_provider, model, api_key, base_url)
            print("LLM client creation: SUCCESS")
            return True
        except Exception as e:
            print(f"LLM client creation: FAILED - {e}")
            return False
    else:
        print("Skipping LLM client creation due to connection failure")
        return False

def test_openai_config():
    """Test OpenAI configuration"""
    print("\nTesting OpenAI Configuration...")
    
    llm_provider = "OpenAI"
    model = "gpt-4o"
    api_key = "your-openai-api-key-here"
    
    print(f"Model: {model}")
    
    try:
        client = get_llm_client(llm_provider, model, api_key)
        print("OpenAI LLM client creation: SUCCESS")
        return True
    except Exception as e:
        print(f"OpenAI LLM client creation: FAILED - {e}")
        return False

def test_ollama_config():
    """Test Ollama configuration"""
    print("\nTesting Ollama Configuration...")
    
    llm_provider = "Ollama"
    model = "llama2"
    base_url = "http://localhost:11434"
    
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    
    try:
        client = get_llm_client(llm_provider, model, ollama_base_url=base_url)
        print("Ollama LLM client creation: SUCCESS")
        return True
    except Exception as e:
        print(f"Ollama LLM client creation: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("LLM Configuration Test Script")
    print("=" * 40)
    
    # Set logging to DEBUG for detailed output
    os.environ['LITELLM_LOG'] = 'DEBUG'
    
    # Test each provider
    results = []
    
    # Uncomment the tests you want to run and add your API keys
    # results.append(test_openai_compatible_config())
    # results.append(test_openai_config())
    # results.append(test_ollama_config())
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    for i, result in enumerate(results):
        status = "PASS" if result else "FAIL"
        print(f"Test {i+1}: {status}")
    
    if all(results):
        print("\nAll tests PASSED! ✅")
    else:
        print("\nSome tests FAILED! ❌")
        print("Please check your configuration and try again.") 