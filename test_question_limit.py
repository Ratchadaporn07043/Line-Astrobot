#!/usr/bin/env python3
"""
Test script for the 3-question limit feature
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_utils import check_and_update_question_limit
from datetime import datetime

def test_question_limit():
    """Test the question limit functionality"""
    test_user_id = "test_user_123"
    
    print("🧪 Testing 3-question limit feature...")
    print("=" * 50)
    
    # Test 1: First question
    print("\n📝 Test 1: First question")
    is_allowed, count, message = check_and_update_question_limit(test_user_id, max_questions=3)
    print(f"Allowed: {is_allowed}")
    print(f"Count: {count}")
    print(f"Message: {message}")
    
    # Test 2: Second question
    print("\n📝 Test 2: Second question")
    is_allowed, count, message = check_and_update_question_limit(test_user_id, max_questions=3)
    print(f"Allowed: {is_allowed}")
    print(f"Count: {count}")
    print(f"Message: {message}")
    
    # Test 3: Third question
    print("\n📝 Test 3: Third question")
    is_allowed, count, message = check_and_update_question_limit(test_user_id, max_questions=3)
    print(f"Allowed: {is_allowed}")
    print(f"Count: {count}")
    print(f"Message: {message}")
    
    # Test 4: Fourth question (should be blocked)
    print("\n📝 Test 4: Fourth question (should be blocked)")
    is_allowed, count, message = check_and_update_question_limit(test_user_id, max_questions=3)
    print(f"Allowed: {is_allowed}")
    print(f"Count: {count}")
    print(f"Message: {message}")
    
    print("\n✅ Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_question_limit()
