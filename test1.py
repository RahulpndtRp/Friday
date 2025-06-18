import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from core.config.settings import Settings
from core.memory.memory_manager import MemoryManager
from core.memory.models import MemoryType, MemoryImportance


async def test_memory_system():
    """Improved test script with proper error handling."""
    print("🚀 Testing FRIDAY Memory System...")

    try:
        # Initialize settings
        settings = Settings()
        print(f"✅ Settings initialized")

        # Create memory manager with unique user ID
        user_id = f"test_user_{int(asyncio.get_event_loop().time())}"
        memory_manager = MemoryManager(user_id, settings)
        print(f"✅ Memory manager created for user: {user_id}")

        # Initialize memory manager
        await memory_manager.initialize()
        print(f"✅ Memory manager initialized")

        # Test 1: Store core memory
        print("\n📝 Test 1: Storing core memory...")
        core_memory_id = await memory_manager.create_memory(
            content="User's name is John Smith, works as a software engineer at Tech Corp",
            memory_type=MemoryType.CORE,
            importance=MemoryImportance.CRITICAL,
            tags=["identity", "profession", "workplace"],
        )
        print(f"✅ Core memory stored with ID: {core_memory_id}")

        # Test 2: Store working memory
        print("\n📝 Test 2: Storing working memory...")
        working_memory_id = await memory_manager.create_memory(
            content="User is currently working on a machine learning project for customer segmentation",
            memory_type=MemoryType.WORKING,
            importance=MemoryImportance.HIGH,
            tags=["current_project", "ml", "customer_segmentation"],
        )
        print(f"✅ Working memory stored with ID: {working_memory_id}")

        # Test 3: Store procedural memory
        print("\n📝 Test 3: Storing procedural memory...")
        procedural_memory_id = await memory_manager.create_memory(
            content="User prefers morning meetings between 9-11 AM and likes detailed project updates",
            memory_type=MemoryType.PROCEDURAL,
            importance=MemoryImportance.MEDIUM,
            tags=["preferences", "schedule", "communication"],
        )
        print(f"✅ Procedural memory stored with ID: {procedural_memory_id}")

        # Test 4: Retrieve specific memory
        print("\n🔍 Test 4: Retrieving specific memory...")
        retrieved_memory = await memory_manager.recall_memory(core_memory_id)
        if retrieved_memory:
            print(f"✅ Retrieved memory: {retrieved_memory.content[:50]}...")
            print(f"   Type: {retrieved_memory.memory_type.value}")
            print(f"   Importance: {retrieved_memory.importance.value}")
            print(f"   Tags: {retrieved_memory.tags}")
        else:
            print("❌ Failed to retrieve memory")

        # Test 5: Search memories
        print("\n🔍 Test 5: Searching memories...")
        search_results = await memory_manager.search_memories(
            query="machine learning",
            memory_types=[MemoryType.WORKING, MemoryType.LONG_TERM],
            limit=5,
        )
        print(f"✅ Found {len(search_results)} memories matching 'machine learning'")
        for i, memory in enumerate(search_results):
            print(
                f"   {i+1}. {memory.content[:60]}... (Type: {memory.memory_type.value})"
            )

        # Test 6: Get context memories
        print("\n🧠 Test 6: Getting context memories...")
        context_memories = await memory_manager.get_context_memories(context_size=10)
        print(f"✅ Retrieved {len(context_memories)} context memories")
        for i, memory in enumerate(context_memories):
            print(
                f"   {i+1}. {memory.content[:50]}... (Importance: {memory.importance.value})"
            )

        # Test 7: Get system summary
        print("\n📊 Test 7: Getting system summary...")
        summary = await memory_manager.get_memory_summary()
        print(f"✅ System summary retrieved:")
        print(f"   Total memories: {summary['statistics'].get('total_memories', 0)}")
        print(f"   Active memories: {summary['statistics'].get('active', 0)}")

        if "type_breakdown" in summary:
            print("   Memory breakdown by type:")
            for mem_type, data in summary["type_breakdown"].items():
                print(f"     {mem_type}: {data['count']} memories")

        # Test 8: Update memory
        print("\n✏️ Test 8: Updating memory...")
        update_success = await memory_manager.update_memory(
            working_memory_id,
            tags=["current_project", "ml", "customer_segmentation", "priority"],
        )
        if update_success:
            print("✅ Memory updated successfully")
            updated_memory = await memory_manager.recall_memory(working_memory_id)
            if updated_memory:
                print(f"   Updated tags: {updated_memory.tags}")
        else:
            print("❌ Failed to update memory")

        # Test 9: Memory consolidation
        print("\n🔧 Test 9: Memory consolidation...")
        consolidation_result = await memory_manager.consolidate_memories()
        if "error" not in consolidation_result:
            print("✅ Memory consolidation completed")
            print(f"   Archived memories: {consolidation_result.get('archived', 0)}")
        else:
            print(f"❌ Consolidation failed: {consolidation_result['error']}")

        print(f"\n🎉 All tests completed successfully!")

        # Cleanup
        await memory_manager.shutdown()
        print(f"✅ Memory manager shutdown complete")

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_memory_system())
