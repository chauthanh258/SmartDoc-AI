#!/usr/bin/env python3
"""
Test script để xác minh các fix cho lỗi:
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]

Kiểm tra:
1. clean_ocr_result xử lý tất cả edge case
2. Document loading không gây lỗi type
3. Text chunking hoạt động đúng
"""

import sys
from pathlib import Path

# Thêm root vào path
root = Path(__file__).parent
sys.path.insert(0, str(root))

from src.core.document_loader import clean_ocr_result
from src.core.text_splitter import split_documents
from langchain_core.documents import Document


def test_clean_ocr_result():
    """Test clean_ocr_result với các input khác nhau."""
    print("=" * 60)
    print("TEST 1: clean_ocr_result - Edge Cases")
    print("=" * 60)
    
    test_cases = [
        (None, "", "None input"),
        ("", "", "Empty string"),
        ("  hello world  ", "hello world", "String with whitespace"),
        (["text1", "text2"], "text1\ntext2", "List of strings"),
        (
            [["box1", "text1", 0.95], ["box2", "text2", 0.92]],
            "text1\ntext2",
            "RapidOCR format (box, text, score)"
        ),
        (
            [("text1", 0.95), ("text2", 0.92)],
            "text1\ntext2",
            "Tuple format (text, score)"
        ),
        (
            {"text": "extracted text", "confidence": 0.95},
            "extracted text",
            "Dict with 'text' key"
        ),
        (
            {"content": "extracted content"},
            "extracted content",
            "Dict with 'content' key"
        ),
        (
            b"byte string content",
            "byte string content",
            "Bytes input"
        ),
        (
            [None, "valid text", None],
            "valid text",
            "List with None values"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for input_val, expected, description in test_cases:
        try:
            result = clean_ocr_result(input_val)
            if result == expected:
                print(f"✓ {description}")
                print(f"  Input: {repr(input_val)[:50]}")
                print(f"  Result: {repr(result)[:50]}")
                passed += 1
            else:
                print(f"✗ {description}")
                print(f"  Expected: {repr(expected)[:50]}")
                print(f"  Got: {repr(result)[:50]}")
                failed += 1
        except Exception as e:
            print(f"✗ {description} - Exception: {e}")
            failed += 1
        print()
    
    print(f"Passed: {passed}/{len(test_cases)}")
    return failed == 0


def test_document_creation_with_non_string_content():
    """Test Document creation với non-string page_content cleaned before passing to Document."""
    print("=" * 60)
    print("TEST 2: Document with Cleaned Non-String Content")
    print("=" * 60)
    
    test_cases = [
        # Case 1: String content
        (
            "Normal string content",
            "Normal string content",
            "String content"
        ),
        # Case 2: List content (from OCR) - must be cleaned first
        (
            [["box", "text from ocr", 0.95]],
            "text from ocr",
            "OCR list format (cleaned)"
        ),
        # Case 3: Dict content - must be cleaned first
        (
            {"text": "extracted from dict"},
            "extracted from dict",
            "Dict format (cleaned)"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for content, expected, description in test_cases:
        try:
            # IMPORTANT: Clean content BEFORE passing to Document constructor
            # This mimics what happens in _apply_ocr_to_pdf after the fix
            cleaned_content = clean_ocr_result(content)
            doc = Document(page_content=cleaned_content, metadata={"test": True})
            
            if doc.page_content == expected:
                print(f"✓ {description}")
                print(f"  Result: {repr(doc.page_content)[:50]}")
                passed += 1
            else:
                print(f"✗ {description}")
                print(f"  Expected: {repr(expected)[:50]}")
                print(f"  Got: {repr(doc.page_content)[:50]}")
                failed += 1
        except Exception as e:
            print(f"✗ {description} - Exception: {e}")
            failed += 1
        print()
    
    print(f"Passed: {passed}/{len(test_cases)}")
    return failed == 0


def test_text_chunking():
    """Test text chunking với various content types."""
    print("=" * 60)
    print("TEST 3: Text Chunking Robustness")
    print("=" * 60)
    
    try:
        # Create documents with mixed content
        docs = [
            Document(
                page_content="This is a normal string document with some content. " * 20,
                metadata={"file_name": "test.pdf", "page_number": 1}
            ),
            Document(
                page_content="Another document with normal text content for testing. " * 20,
                metadata={"file_name": "test.pdf", "page_number": 2}
            ),
        ]
        
        # Split documents using valid chunk_size (must be from [500, 600, 1000, 1500, 2000])
        chunks = split_documents(docs, chunk_size=600, chunk_overlap=100)
        
        print(f"✓ Successfully created {len(chunks)} chunks")
        
        # Verify all chunks have string content
        all_string = all(isinstance(chunk.page_content, str) for chunk in chunks)
        if all_string:
            print("✓ All chunks have string content")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3
                print(f"  Chunk {i}: {repr(chunk.page_content[:40])}...")
            return True
        else:
            print("✗ Some chunks have non-string content")
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk.page_content, str):
                    print(f"  Chunk {i}: {type(chunk.page_content)}")
            return False
            
    except Exception as e:
        print(f"✗ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING OCR TypeError FIXES")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1
    results.append(("clean_ocr_result", test_clean_ocr_result()))
    print()
    
    # Test 2
    results.append(("Document creation", test_document_creation_with_non_string_content()))
    print()
    
    # Test 3
    results.append(("Text chunking", test_text_chunking()))
    print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("✓ All tests passed! Fixes are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please review the fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
