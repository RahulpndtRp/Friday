#!/usr/bin/env python3
"""
Debug script to identify and fix FRIDAY's memory search issues
Analyzes vector embeddings, FAISS index, and search mechanics
"""

import asyncio
import os
import numpy as np
from datetime import datetime


async def debug_memory_search():
    """Comprehensive debugging of memory search functionality"""
    print("🔍 FRIDAY Memory Search Diagnostic")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False

    try:
        from my_mem.memory.main import AsyncMemory
        from my_mem.configs.base import MemoryConfig

        # Initialize memory system
        config = MemoryConfig(llm={"provider": "openai_async", "config": {}})
        memory = AsyncMemory(config)
        user_id = "debug_user"

        print("🧪 Test 1: Vector Embedding Quality")
        print("-" * 30)

        # Test diverse content to ensure different embeddings
        test_content = [
            "I love pizza and Italian food",
            "I work as a software engineer in Python",
            "I live in New York City",
            "I prefer morning workouts at the gym",
            "My favorite color is blue and I drive a Tesla",
        ]

        embeddings = []
        for i, content in enumerate(test_content):
            embedding = memory.embedder.embed(content, "add")
            embeddings.append(embedding)
            print(f"   📊 Content {i+1}: {content[:40]}...")
            print(f"      🔢 Embedding shape: {embedding.shape}")
            print(f"      📈 First 5 values: {embedding[:5]}")
            print(
                f"      📊 Mean: {np.mean(embedding):.6f}, Std: {np.std(embedding):.6f}"
            )

        # Check embedding diversity
        print("\n🔬 Embedding Similarity Analysis:")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"   📏 Content {i+1} vs {j+1}: {similarity:.4f} similarity")

        print("\n🧪 Test 2: FAISS Index Health")
        print("-" * 30)

        # Clear any existing data
        try:
            await memory.reset()
            print("   ✅ Memory reset successful")
        except:
            print("   ⚠️  Memory reset not available, continuing...")

        # Add test memories one by one
        stored_ids = []
        for i, content in enumerate(test_content):
            result = await memory.add(
                content, user_id=user_id, infer=False
            )  # No inference for clean test
            if result.get("results"):
                stored_ids.append(result["results"][0]["id"])
                print(f"   ✅ Stored: '{content[:40]}...' -> ID: {stored_ids[-1][:8]}")
            else:
                print(f"   ❌ Failed to store: '{content[:40]}...'")

        # Check FAISS index state
        vs = memory.vector_store
        if hasattr(vs, "_index"):
            print(f"   📊 FAISS index size: {vs._index.ntotal}")
            print(f"   📦 Payload count: {len(vs._payloads)}")
            print(f"   🔧 Index type: {type(vs._index)}")

        print("\n🧪 Test 3: Search Mechanics")
        print("-" * 30)

        # Test search for each stored item
        search_queries = [
            ("food preferences", "pizza"),
            ("programming work", "Python"),
            ("location", "New York"),
            ("exercise habits", "gym"),
            ("personal items", "Tesla"),
        ]

        for query_desc, query in search_queries:
            print(f"\n   🔍 Testing: {query_desc} -> '{query}'")

            # Get query embedding
            query_embedding = memory.embedder.embed(query, "search")
            print(f"      🔢 Query embedding shape: {query_embedding.shape}")

            # Search memories
            results = await memory.search(query, user_id=user_id, limit=5)
            memories = results.get("results", [])

            print(f"      📊 Found {len(memories)} results:")
            for i, mem in enumerate(memories):
                print(
                    f"         {i+1}. Score: {mem['score']:.4f} - '{mem['memory'][:50]}...'"
                )

            # Direct FAISS search for comparison
            if hasattr(vs, "search"):
                direct_results = vs.search(
                    query=query,
                    vectors=query_embedding,
                    limit=5,
                    filters={"user_id": user_id},
                )
                print(f"      🔧 Direct FAISS results: {len(direct_results)}")
                for i, hit in enumerate(direct_results[:3]):
                    print(
                        f"         Direct {i+1}. Score: {hit.score:.4f} - '{hit.payload.get('data', '')[:30]}...'"
                    )

        print("\n🧪 Test 4: Score Distribution Analysis")
        print("-" * 30)

        # Analyze if all scores are suspiciously similar
        all_scores = []
        for query_desc, query in search_queries:
            results = await memory.search(query, user_id=user_id, limit=5)
            scores = [mem["score"] for mem in results.get("results", [])]
            all_scores.extend(scores)
            print(f"   📊 '{query}' scores: {[f'{s:.4f}' for s in scores[:3]]}")

        if all_scores:
            print(f"   📈 Score statistics:")
            print(f"      Min: {min(all_scores):.4f}")
            print(f"      Max: {max(all_scores):.4f}")
            print(f"      Mean: {np.mean(all_scores):.4f}")
            print(f"      Std: {np.std(all_scores):.6f}")

            # Check if all scores are too similar (potential bug)
            if np.std(all_scores) < 0.001:
                print("   ⚠️  WARNING: All scores very similar - possible search bug!")

        print("\n🧪 Test 5: Memory Persistence Check")
        print("-" * 30)

        # Check if memories persist and are retrievable
        all_memories = await memory.get_all(user_id)
        stored_memories = all_memories.get("results", [])

        print(f"   📊 Total stored memories: {len(stored_memories)}")
        for i, mem in enumerate(stored_memories[:5]):
            print(f"      {i+1}. ID: {mem['id'][:8]} - '{mem['memory'][:40]}...'")

        # Test specific ID retrieval
        if stored_memories:
            test_id = stored_memories[0]["id"]
            # Note: Direct get by ID might not be available in current interface
            print(f"   🎯 Testing retrieval of specific ID: {test_id[:8]}")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False


async def suggest_fixes():
    """Suggest potential fixes based on common issues"""
    print("\n🔧 POTENTIAL FIXES TO TRY")
    print("=" * 50)

    fixes = [
        {
            "issue": "All embeddings too similar",
            "fix": "Check OpenAI API key, model version, or embedding normalization",
            "test": "Compare embeddings of very different content",
        },
        {
            "issue": "FAISS index corruption",
            "fix": "Reset FAISS index and rebuild from scratch",
            "test": "Delete .faiss directory and restart",
        },
        {
            "issue": "Score calculation bug",
            "fix": "Check if using cosine similarity vs dot product correctly",
            "test": "Verify FAISS metric type matches expectation",
        },
        {
            "issue": "User ID filtering issue",
            "fix": "Verify user_id filter is working in FAISS search",
            "test": "Add memories for different users and check isolation",
        },
        {
            "issue": "Vector dimensionality mismatch",
            "fix": "Ensure embedding dims match FAISS index dims",
            "test": "Check embedding.shape vs FAISS index dimensions",
        },
    ]

    for i, fix in enumerate(fixes, 1):
        print(f"{i}. 🎯 {fix['issue']}")
        print(f"   💡 Fix: {fix['fix']}")
        print(f"   🧪 Test: {fix['test']}")
        print()

    print("🚀 Quick Fix Command:")
    print("   rm -rf .faiss/  # Reset FAISS index")
    print("   python debug_memory_search.py  # Re-run this script")


async def main():
    """Run memory search debugging"""
    success = await debug_memory_search()
    await suggest_fixes()

    if success:
        print("\n✅ Debugging completed successfully")
        print("📋 Next: Apply suggested fixes and re-test")
    else:
        print("\n❌ Debugging failed - check error messages above")


if __name__ == "__main__":
    asyncio.run(main())
