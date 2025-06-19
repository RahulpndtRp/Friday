#!/usr/bin/env python3
"""
Quick test to verify memrp integration with FRIDAY
Run this after copying the memrp files
"""

import os
import asyncio


def test_imports():
    """Test if all memrp components can be imported"""
    print("🧪 Testing memrp imports...")

    try:
        from my_mem.memory.main import AsyncMemory

        print("✅ AsyncMemory imported")
    except ImportError as e:
        print(f"❌ AsyncMemory import failed: {e}")
        return False

    try:
        from my_mem.rag.rag_pipeline import AsyncRAGPipeline

        print("✅ AsyncRAGPipeline imported")
    except ImportError as e:
        print(f"❌ AsyncRAGPipeline import failed: {e}")
        return False

    try:
        from my_mem.configs.base import MemoryConfig

        print("✅ MemoryConfig imported")
    except ImportError as e:
        print(f"❌ MemoryConfig import failed: {e}")
        return False

    return True


async def test_memory_basic():
    """Test basic memory operations"""
    print("\n🧠 Testing basic memory operations...")

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it first.")
        return False

    try:
        from my_mem.memory.main import AsyncMemory
        from my_mem.configs.base import MemoryConfig

        # Create memory config
        config = MemoryConfig()
        memory = AsyncMemory(config)

        # Test adding memory
        result = await memory.add(
            "I love pizza and prefer thin crust", user_id="test_user", infer=True
        )
        print(f"✅ Memory add result: {len(result.get('results', []))} items stored")

        # Test searching memory
        search_result = await memory.search(
            "what food do I like", user_id="test_user", limit=3
        )
        memories = search_result.get("results", [])
        print(f"✅ Memory search found: {len(memories)} results")

        if memories:
            print(f"   First result: {memories[0].get('memory', '')[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


async def test_rag():
    """Test RAG functionality"""
    print("\n🤖 Testing RAG functionality...")

    try:
        from my_mem.memory.main import AsyncMemory
        from my_mem.rag.rag_pipeline import AsyncRAGPipeline
        from my_mem.configs.base import MemoryConfig

        config = MemoryConfig()
        memory = AsyncMemory(config)
        rag = AsyncRAGPipeline(memory)

        # Add some test data first
        await memory.add(
            "I work as a software engineer and love Python programming",
            user_id="test_user",
            infer=True,
        )

        # Test RAG query
        result = await rag.query("What do I do for work?", user_id="test_user")

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"✅ RAG query successful")
        print(f"   Answer: {answer[:100]}...")
        print(f"   Sources: {len(sources)} references")

        return True

    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False


def test_friday_integration():
    """Test if FRIDAY can import the enhanced chat manager"""
    print("\n🤖 Testing FRIDAY integration...")

    try:
        # Test if we can import from FRIDAY's chat manager
        import sys

        sys.path.append("src")

        from core.chat.chat_manager import EnhancedChatManager

        print("✅ EnhancedChatManager imported from FRIDAY")

        # Test if it has memrp integration
        if hasattr(EnhancedChatManager, "_initialize_memory_system"):
            print("✅ Memory system integration found")
        else:
            print("❌ Memory system integration not found")
            return False

        return True

    except ImportError as e:
        print(f"❌ FRIDAY integration test failed: {e}")
        print("   Make sure you've replaced chat_manager.py with the enhanced version")
        return False


async def main():
    """Run all tests"""
    print("🧠 FRIDAY Memory Integration Test")
    print("=================================")

    all_passed = True

    # Test 1: Import test
    if not test_imports():
        print("\n❌ Import test failed. Make sure you've copied all memrp files.")
        all_passed = False

    # Test 2: FRIDAY integration
    if not test_friday_integration():
        print("\n❌ FRIDAY integration test failed.")
        all_passed = False

    # Test 3: Basic memory (only if imports work)
    if all_passed:
        if not await test_memory_basic():
            print("\n❌ Basic memory test failed.")
            all_passed = False

    # Test 4: RAG test (only if basic memory works)
    if all_passed:
        if not await test_rag():
            print("\n❌ RAG test failed.")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your FRIDAY memory system is ready.")
        print("\n📋 Next steps:")
        print("1. Start FRIDAY: python main.py")
        print("2. Test memory: 'I love Italian food'")
        print("3. Test recall: 'What food do I like?'")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all memrp files are copied correctly")
        print("2. Install dependencies: pip install numpy faiss-cpu pytz")
        print("3. Set OPENAI_API_KEY environment variable")
        print("4. Replace chat_manager.py with enhanced version")


if __name__ == "__main__":
    asyncio.run(main())
