#!/usr/bin/env python3
"""
Test script to verify the error fixes in retrieval_utils.py
"""
import os
import sys
from dotenv import load_dotenv

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_configuration():
    """Test if configuration is properly set up"""
    print("🔍 Testing configuration...")
    
    # Load .env if it exists
    if os.path.exists('.env'):
        load_dotenv()
        print("✅ .env file found and loaded")
    else:
        print("⚠️ .env file not found")
    
    # Check environment variables
    mongo_url = os.getenv("MONGO_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"MONGO_URL configured: {'✅' if mongo_url and mongo_url != 'mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority' else '❌'}")
    print(f"OPENAI_API_KEY configured: {'✅' if openai_key and openai_key != 'your-openai-api-key-here' else '❌'}")
    
    return mongo_url and openai_key and mongo_url != 'mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority' and openai_key != 'your-openai-api-key-here'

def test_import():
    """Test if modules can be imported"""
    print("\n🔍 Testing imports...")
    try:
        from app.retrieval_utils import ask_question_to_rag
        print("✅ retrieval_utils imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid configuration"""
    print("\n🔍 Testing error handling...")
    try:
        from app.retrieval_utils import ask_question_to_rag
        
        # This should return an error message instead of hanging
        result = ask_question_to_rag("ทดสอบ", "test_user")
        print(f"✅ Function completed without hanging")
        print(f"📝 Result: {result[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing AstroBot Error Fixes")
    print("=" * 60)
    
    config_ok = test_configuration()
    import_ok = test_import()
    error_handling_ok = test_error_handling()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"Import: {'✅' if import_ok else '❌'}")
    print(f"Error Handling: {'✅' if error_handling_ok else '❌'}")
    
    if not config_ok:
        print("\n📝 To fix configuration issues:")
        print("1. Create a .env file in the project root")
        print("2. Add your MongoDB connection string: MONGO_URL=your_connection_string")
        print("3. Add your OpenAI API key: OPENAI_API_KEY=your_api_key")
    
    print("=" * 60)
