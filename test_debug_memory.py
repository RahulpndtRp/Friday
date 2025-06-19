# ============================================================================
# FRIDAY MEMORY DIAGNOSTIC TOOL
# ============================================================================

# Add this as a new file: debug_memory.py

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config.settings import Settings
from src.core.memory.memory_manager import MemoryManager
from src.core.memory.models import MemoryType, MemoryImportance
from src.core.chat.chat_manager import ChatManager
from src.core.chat.chat_store import ChatStore


async def diagnose_memory_system():
    """Comprehensive memory system diagnosis."""
    print("🔍 FRIDAY Memory System Diagnostic")
    print("=" * 50)

    settings = Settings()
    user_id = "friday_user_001"

    try:
        # 1. Check Memory Manager
        print("\n📊 MEMORY MANAGER DIAGNOSTICS:")
        memory_manager = MemoryManager(user_id, settings)
        await memory_manager.initialize()

        # Get memory statistics
        summary = await memory_manager.get_memory_summary()
        print(f"✅ Memory Manager Status: Initialized")
        print(f"📈 Total Memories: {summary['statistics'].get('total_memories', 0)}")
        print(f"🔴 Active Memories: {summary['statistics'].get('active', 0)}")
        print(f"📦 Archived Memories: {summary['statistics'].get('archived', 0)}")

        # Check memory types
        print(f"\n📋 Memory Breakdown by Type:")
        for mem_type, data in summary.get("type_breakdown", {}).items():
            print(f"   {mem_type}: {data['count']} memories")

        # 2. Check Chat Store
        print(f"\n💬 CHAT STORE DIAGNOSTICS:")
        chat_store = ChatStore(user_id, settings)
        await chat_store.initialize()

        # Get recent conversations
        conversations = await chat_store.get_user_conversations(limit=5)
        print(f"✅ Chat Store Status: Initialized")
        print(f"📝 Total Conversations: {len(conversations)}")

        if conversations:
            latest_conv = conversations[0]
            print(f"🕒 Latest Conversation: {latest_conv.id}")
            print(f"📅 Last Activity: {latest_conv.last_activity}")
            print(f"💬 Message Count: {latest_conv.message_count}")

            # Get messages from latest conversation
            messages = await chat_store.get_conversation_messages(
                latest_conv.id, limit=10
            )
            print(f"📨 Recent Messages: {len(messages)}")

            for i, msg in enumerate(messages[-5:]):  # Last 5 messages
                content_preview = (
                    msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                )
                print(f"   {i+1}. [{msg.message_type.value}] {content_preview}")

        # 3. Search for mango-related memories
        print(f"\n🥭 SEARCHING FOR MANGO-RELATED MEMORIES:")
        mango_memories = await memory_manager.search_memories("mango", limit=10)
        print(f"🔍 Found {len(mango_memories)} mango-related memories:")

        for i, mem in enumerate(mango_memories):
            print(f"   {i+1}. [{mem.memory_type.value}] {mem.content[:80]}...")
            print(f"       Tags: {mem.tags}")
            print(f"       Importance: {mem.importance.value}")
            print(f"       Created: {mem.created_at}")

        # 4. Search for recipe-related memories
        print(f"\n🍳 SEARCHING FOR RECIPE-RELATED MEMORIES:")
        recipe_memories = await memory_manager.search_memories("recipe", limit=10)
        print(f"🔍 Found {len(recipe_memories)} recipe-related memories:")

        for i, mem in enumerate(recipe_memories):
            print(f"   {i+1}. [{mem.memory_type.value}] {mem.content[:80]}...")

        # 5. Check all recent memories
        print(f"\n⏰ RECENT MEMORIES (Last 24 hours):")
        recent_memories = await memory_manager.primary_store.get_recent_memories(
            hours=24, limit=15
        )
        print(f"🔍 Found {len(recent_memories)} recent memories:")

        for i, mem in enumerate(recent_memories):
            print(f"   {i+1}. [{mem.memory_type.value}] {mem.content[:80]}...")
            print(f"       Created: {mem.created_at}")

        # 6. Test memory storage
        print(f"\n🧪 TESTING MEMORY STORAGE:")
        test_memory_id = await memory_manager.create_memory(
            content="Test memory: User mentioned they like mangoes in diagnostic test",
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.MEDIUM,
            tags=["test", "mango", "preference"],
        )
        print(f"✅ Test memory created: {test_memory_id}")

        # Retrieve the test memory
        retrieved = await memory_manager.recall_memory(test_memory_id)
        if retrieved:
            print(f"✅ Test memory retrieved successfully")
        else:
            print(f"❌ Failed to retrieve test memory")

        # 7. Integration test - simulate chat context building
        print(f"\n🔗 TESTING CHAT-MEMORY INTEGRATION:")
        chat_manager = ChatManager(user_id, settings)
        await chat_manager.initialize()

        # Set up dependencies
        chat_manager.memory_manager = memory_manager

        # Test context building
        test_context = await chat_manager._build_context(
            "What can I make with mango?",
            latest_conv.id if conversations else "test_conv",
        )

        print(f"🔍 Context Built:")
        print(
            f"   Memory contexts found: {len(test_context.get('memory_context', []))}"
        )
        print(
            f"   Document contexts found: {len(test_context.get('document_context', []))}"
        )
        print(f"   Sources found: {len(test_context.get('sources', []))}")

        if test_context.get("memory_context"):
            print(f"   Memory context preview:")
            for i, mem_ctx in enumerate(test_context["memory_context"][:3]):
                print(f"     {i+1}. {mem_ctx.get('content', '')[:60]}...")

        # Cleanup
        await memory_manager.shutdown()
        await chat_store.shutdown()

        print(f"\n✅ Diagnostic completed successfully!")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")


async def check_database_files():
    """Check if database files exist and their sizes."""
    print(f"\n💾 DATABASE FILES CHECK:")

    settings = Settings()
    user_id = "friday_user_001"

    # Check memory database
    memory_db_path = f"{settings.memory_db_path}_{user_id}.db"
    chat_db_path = f"{settings.memory_db_path}_chat_{user_id}.db"
    documents_db_path = f"{settings.memory_db_path}_documents_{user_id}.db"

    for db_name, db_path in [
        ("Memory DB", memory_db_path),
        ("Chat DB", chat_db_path),
        ("Documents DB", documents_db_path),
    ]:
        path_obj = Path(db_path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            print(f"✅ {db_name}: {db_path} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {db_name}: {db_path} (NOT FOUND)")


if __name__ == "__main__":
    print("🤖 Starting FRIDAY Memory Diagnostic...")
    asyncio.run(check_database_files())
    asyncio.run(diagnose_memory_system())
