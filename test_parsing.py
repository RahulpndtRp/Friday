import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from core.config.settings import Settings
from core.documents.document_manager import DocumentManager
from core.documents.models import DocumentCategory, DocumentType, ProcessingStatus


async def test_document_processing():
    """Test document processing system."""
    print("🚀 Testing FRIDAY Document Processing System...")

    try:
        # Initialize settings
        settings = Settings()
        print(f"✅ Settings initialized")

        # Create document manager
        user_id = f"test_user_{int(asyncio.get_event_loop().time())}"
        doc_manager = DocumentManager(user_id, settings)
        print(f"✅ Document manager created for user: {user_id}")

        # Initialize document manager
        await doc_manager.initialize()
        print(f"✅ Document manager initialized")

        # Test 1: Create test documents
        print("\n📝 Test 1: Creating test documents...")

        # Create sample text file
        test_dir = Path("test_documents")
        test_dir.mkdir(exist_ok=True)

        sample_text = """
# Sample Document for Testing

This is a test document for the FRIDAY document processing system.

## Introduction
This document contains various sections to test the chunking and processing capabilities.

## Features
The document processing system can:
- Parse PDF, DOCX, and TXT files
- Extract metadata and content
- Create intelligent chunks
- Store documents in database
- Enable search and retrieval

## Conclusion
This is a comprehensive document processing system for personal AI assistants.
"""

        text_file = test_dir / "sample_document.txt"
        with open(text_file, "w") as f:
            f.write(sample_text)

        print(f"✅ Created test document: {text_file}")

        # Test 2: Process the document
        print("\n📄 Test 2: Processing document...")
        try:
            processed_doc = await doc_manager.process_document(
                file_path=str(text_file),
                category=DocumentCategory.REFERENCE,
                tags=["test", "sample", "documentation"],
                metadata={"source": "test_system", "version": "1.0"},
            )

            print(f"✅ Document processed successfully!")
            print(f"   Document ID: {processed_doc.id}")
            print(f"   Title: {processed_doc.title}")
            print(f"   Type: {processed_doc.document_type.value}")
            print(f"   Category: {processed_doc.category.value}")
            print(f"   Status: {processed_doc.status.value}")
            print(f"   Chunks: {processed_doc.chunk_count}")
            print(f"   Word count: {processed_doc.word_count}")
            print(f"   Processing time: {processed_doc.processing_time:.3f}s")

        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            return

        # Test 3: Retrieve the document
        print("\n🔍 Test 3: Retrieving document...")
        retrieved_doc = await doc_manager.get_document(processed_doc.id)
        if retrieved_doc:
            print(f"✅ Document retrieved successfully")
            print(f"   Retrieved {len(retrieved_doc.chunks)} chunks")
            print(f"   First chunk preview: {retrieved_doc.chunks[0].content[:100]}...")
        else:
            print(f"❌ Failed to retrieve document")

        # Test 4: Search documents
        print("\n🔍 Test 4: Searching documents...")
        search_results = await doc_manager.search_documents(
            query="document processing",
            categories=[DocumentCategory.REFERENCE],
            limit=5,
        )
        print(
            f"✅ Found {len(search_results)} documents matching 'document processing'"
        )
        for i, doc in enumerate(search_results):
            print(
                f"   {i+1}. {doc.title} ({doc.document_type.value}) - {doc.word_count} words"
            )

        # Test 5: Search content chunks
        print("\n🔍 Test 5: Searching content chunks...")
        chunk_results = await doc_manager.search_content(query="features", limit=10)
        print(f"✅ Found {len(chunk_results)} chunks matching 'features'")
        for i, chunk in enumerate(chunk_results):
            print(
                f"   {i+1}. Type: {chunk.chunk_type.value}, Words: {chunk.word_count}"
            )
            print(f"      Content: {chunk.content[:80]}...")

        # Test 6: List user documents
        print("\n📋 Test 6: Listing user documents...")
        user_docs = await doc_manager.get_user_documents(
            status=ProcessingStatus.COMPLETED, limit=10
        )
        print(f"✅ User has {len(user_docs)} completed documents")
        for i, doc in enumerate(user_docs):
            print(
                f"   {i+1}. {doc.title} - {doc.category.value} - {doc.chunk_count} chunks"
            )

        # Test 7: Get statistics
        print("\n📊 Test 7: Getting document statistics...")
        stats = await doc_manager.get_document_statistics()
        print(f"✅ Document statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total words: {stats.get('total_words', 0)}")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(
            f"   Average processing time: {stats.get('average_processing_time', 0):.3f}s"
        )

        if "by_type" in stats:
            print("   Documents by type:")
            for doc_type, count in stats["by_type"].items():
                print(f"     {doc_type}: {count}")

        if "by_category" in stats:
            print("   Documents by category:")
            for category, count in stats["by_category"].items():
                print(f"     {category}: {count}")

        print(f"\n🎉 All document processing tests completed successfully!")

        # Cleanup
        await doc_manager.shutdown()
        print(f"✅ Document manager shutdown complete")

        # Clean up test files
        if text_file.exists():
            text_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()
        print(f"✅ Test files cleaned up")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_document_processing())
