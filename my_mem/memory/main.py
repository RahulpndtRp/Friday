"""
Complete memory implementation with both sync and async classes
Replace your my_mem/memory/main.py with this file
"""

import asyncio
import concurrent.futures, hashlib, json, logging, os, uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

from my_mem.configs.base import MemoryConfig, MemoryItem
from my_mem.utils.telemetry import capture_event
from my_mem.utils.prompts import (
    FACT_RETRIEVAL_PROMPT,
    get_update_memory_messages,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
)
from my_mem.utils.utils import (
    parse_messages,
    remove_code_blocks,
)

from my_mem.vector_stores.base import BaseVectorStore
from my_mem.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
from my_mem.memory.storage_sqlite import SQLiteManager
from my_mem.memory.short_memory import ShortTermMemory

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Memory (sync) - for compatibility with RAG pipeline                       #
# --------------------------------------------------------------------------- #
class Memory:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.cfg = config

        # Build provider instances
        self.embedder = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config, config.vector_store.config
        )
        self.vector_store: BaseVectorStore = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )
        self.llm = LlmFactory.create(config.llm.provider, config.llm.config)
        self.db = SQLiteManager(config.history_db_path)
        self.short_term = ShortTermMemory(max_items=32)

        capture_event("memory.init", self, {"sync_type": "sync"})

    def add(self, message: str, *, user_id: str, infer: bool = True) -> Dict:
        """Add message to memory (sync version)"""

        # Always add to short-term memory
        msg_vec = self.embedder.embed(message, "add")
        self.short_term.add(user_id, message, msg_vec)

        metadata = {"user_id": user_id}
        filters = {"user_id": user_id}

        # Simple path - no inference
        if not infer:
            mem_id = self._create_memory(message, msg_vec, metadata)
            return {"results": [{"id": mem_id, "memory": message, "event": "ADD"}]}

        # Fact extraction
        try:
            system_prompt, user_prompt = FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"
            resp = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            facts = json.loads(remove_code_blocks(resp))["facts"]
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            facts = []

        if not facts:
            return {"results": []}

        # Check overlap with existing memories
        existing = {}
        for fact in facts:
            vec = self.embedder.embed(fact, "add")
            hits = self.vector_store.search(fact, vec, 5, filters)
            for h in hits:
                existing[h.id] = h.payload["data"]

        # Get update instructions
        old_mem_list = [{"id": k, "text": v} for k, v in existing.items()]
        up_prompt = get_update_memory_messages(
            old_mem_list, facts, self.cfg.custom_update_memory_prompt
        )

        try:
            resp = self.llm.generate_response(
                messages=[{"role": "user", "content": up_prompt}],
                response_format={"type": "json_object"},
            )
            actions = json.loads(remove_code_blocks(resp))["memory"]
        except Exception as e:
            logger.error(f"LLM update-memory failed: {e}")
            actions = [
                {"id": str(uuid.uuid4()), "text": f, "event": "ADD"} for f in facts
            ]

        # Apply actions
        results = []
        for act in actions:
            ev = act["event"]
            if ev == "ADD":
                vec = self.embedder.embed(act["text"], "add")
                mid = self._create_memory(act["text"], vec, deepcopy(metadata))
                results.append({"id": mid, "memory": act["text"], "event": "ADD"})
            elif ev == "UPDATE":
                vec = self.embedder.embed(act["text"], "update")
                self._update_memory(act["id"], act["text"], vec, deepcopy(metadata))
                results.append(
                    {
                        "id": act["id"],
                        "memory": act["text"],
                        "event": "UPDATE",
                        "previous_memory": act.get("old_memory"),
                    }
                )
            elif ev == "DELETE":
                self._delete_memory(act["id"])
                results.append(
                    {"id": act["id"], "memory": act["text"], "event": "DELETE"}
                )
            else:
                results.append(
                    {"id": act["id"], "memory": act["text"], "event": "NONE"}
                )

        capture_event("memory.add", self, {"facts": len(facts)})
        return {"results": results}

    def search(
        self, query: str, *, user_id: str, limit: int = 5, ltm_threshold: float = 0.75
    ) -> Dict:
        import numpy as np

        filters = {"user_id": user_id}
        qvec = self.embedder.embed(query, "search")

        # ---- LTM: vector store hits over threshold ----
        lt_hits = self.vector_store.search(
            query=query, vectors=qvec, limit=10, filters=filters
        )
        lt_items = [
            MemoryItem(
                id=h.id,
                memory=h.payload["data"],
                hash=h.payload.get("hash"),
                created_at=h.payload.get("created_at"),
                updated_at=h.payload.get("updated_at"),
                score=h.score,  # ✅ USE ACTUAL FAISS SCORE - DON'T OVERRIDE!
            ).model_dump()
            for h in lt_hits
            if h.score >= ltm_threshold  # ✅ APPLY THRESHOLD CORRECTLY
        ][
            :3
        ]  # Take top 3 above threshold

        # ---- STM: last 5 turns only ----
        st_buf = self.short_term.recent(user_id, limit=32)
        st_items = [
            MemoryItem(
                id=it["id"],
                memory=it["text"],
                hash=None,
                created_at=it["created"],
                updated_at=None,
                score=float(0.95),  # ✅ FIXED: Use reasonable STM score, not 0.99
            ).model_dump()
            for it in st_buf[-5:]
        ]

        # ---- Merge and return ----
        merged = sorted(lt_items + st_items, key=lambda x: x["score"], reverse=True)
        return {"results": merged[:limit]}

    def _create_memory(self, data: str, vec, meta: Dict) -> str:
        """Create memory (sync version)"""
        mid = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update(
            {
                "data": data,
                "hash": hashlib.md5(data.encode()).hexdigest(),
                "created_at": now,
                "__vector": vec,
            }
        )
        self.vector_store.insert([vec], [meta], [mid])
        self.db.add_history(mid, None, data, "ADD", created_at=now)
        return mid

    def _update_memory(self, mid, new_data, vec, meta):
        """Update memory (sync version)"""
        existing = self.vector_store.search("", vec, 1, {})
        created = existing[0].payload.get("created_at") if existing else None
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update(
            {
                "data": new_data,
                "hash": hashlib.md5(new_data.encode()).hexdigest(),
                "created_at": created,
                "updated_at": now,
                "__vector": vec,
            }
        )
        self.vector_store.update(mid, vec, meta)
        self.db.add_history(
            mid,
            existing[0].payload["data"] if existing else None,
            new_data,
            "UPDATE",
            created_at=created,
            updated_at=now,
        )

    def _delete_memory(self, mid):
        """Delete memory (sync version)"""
        self.vector_store.delete(mid)
        self.db.add_history(mid, None, None, "DELETE", is_deleted=1)

    def reset(self):
        """Reset memory system"""
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()
        else:
            raise NotImplementedError(
                "The current vector store does not support reset()."
            )


# --------------------------------------------------------------------------- #
#  AsyncMemory - for FRIDAY integration                                       #
# --------------------------------------------------------------------------- #
class AsyncMemory:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.cfg = config

        # Build provider instances
        self.embedder = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config, config.vector_store.config
        )
        self.vector_store: BaseVectorStore = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )

        # FIXED: Use async LLM for AsyncMemory
        if config.llm.provider == "openai":
            # Force async provider for AsyncMemory
            async_config = config.llm.model_copy()
            async_config.provider = "openai_async"
            self.llm = LlmFactory.create(async_config.provider, async_config.config)
        else:
            self.llm = LlmFactory.create(config.llm.provider, config.llm.config)

        self.db = SQLiteManager(config.history_db_path)
        self.short_term = ShortTermMemory(max_items=32)

        capture_event("memory.init", self, {"sync_type": "async"})

    async def add(self, message: str, *, user_id: str, infer: bool = True) -> Dict:
        """Add message to memory with proper async handling"""

        # Always add to short-term memory
        msg_vec = await asyncio.to_thread(self.embedder.embed, message, "add")
        self.short_term.add(user_id, message, msg_vec)

        metadata = {"user_id": user_id}
        filters = {"user_id": user_id}

        # Simple path - no inference
        if not infer:
            mem_id = await self._create_memory(message, msg_vec, metadata)
            return {"results": [{"id": mem_id, "memory": message, "event": "ADD"}]}

        # Fact extraction with proper async handling
        try:
            system_prompt, user_prompt = FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"

            # FIXED: Proper async call
            resp = await self.llm.generate_response_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            facts = json.loads(remove_code_blocks(resp))["facts"]
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            facts = []

        if not facts:
            return {"results": []}

        # Check overlap with existing memories
        existing = {}
        for fact in facts:
            vec = await asyncio.to_thread(self.embedder.embed, fact, "add")
            hits = await asyncio.to_thread(
                self.vector_store.search, fact, vec, 5, filters
            )
            for h in hits:
                existing[h.id] = h.payload["data"]

        # Get update instructions
        old_mem_list = [{"id": k, "text": v} for k, v in existing.items()]
        up_prompt = get_update_memory_messages(
            old_mem_list, facts, self.cfg.custom_update_memory_prompt
        )

        try:
            # FIXED: Proper async call
            resp = await self.llm.generate_response_async(
                messages=[{"role": "user", "content": up_prompt}],
                response_format={"type": "json_object"},
            )
            actions = json.loads(remove_code_blocks(resp))["memory"]
        except Exception as e:
            logger.error(f"LLM update-memory failed: {e}")
            actions = [
                {"id": str(uuid.uuid4()), "text": f, "event": "ADD"} for f in facts
            ]

        # Apply actions
        results = []
        for act in actions:
            ev = act["event"]
            if ev == "ADD":
                vec = await asyncio.to_thread(self.embedder.embed, act["text"], "add")
                mid = await self._create_memory(act["text"], vec, deepcopy(metadata))
                results.append({"id": mid, "memory": act["text"], "event": "ADD"})
            elif ev == "UPDATE":
                vec = await asyncio.to_thread(
                    self.embedder.embed, act["text"], "update"
                )
                await self._update_memory(
                    act["id"], act["text"], vec, deepcopy(metadata)
                )
                results.append(
                    {
                        "id": act["id"],
                        "memory": act["text"],
                        "event": "UPDATE",
                        "previous_memory": act.get("old_memory"),
                    }
                )
            elif ev == "DELETE":
                await self._delete_memory(act["id"])
                results.append(
                    {"id": act["id"], "memory": act["text"], "event": "DELETE"}
                )
            else:
                results.append(
                    {"id": act["id"], "memory": act["text"], "event": "NONE"}
                )

        capture_event("memory.add", self, {"facts": len(facts)})
        return {"results": results}

    # ✅ ASYNC VERSION (same fix)
    async def search(
        self, query: str, *, user_id: str, limit: int = 5, ltm_threshold: float = 0.75
    ) -> Dict:
        import numpy as np

        filters = {"user_id": user_id}
        qvec = await asyncio.to_thread(self.embedder.embed, query, "search")

        lt_hits = await asyncio.to_thread(
            self.vector_store.search, query, qvec, 10, filters
        )
        lt_items = [
            MemoryItem(
                id=h.id,
                memory=h.payload["data"],
                hash=h.payload.get("hash"),
                created_at=h.payload.get("created_at"),
                updated_at=h.payload.get("updated_at"),
                score=h.score,  # ✅ USE ACTUAL FAISS SCORE
            ).model_dump()
            for h in lt_hits
            if h.score >= ltm_threshold  # ✅ APPLY THRESHOLD CORRECTLY
        ][:3]

        st_buf = self.short_term.recent(user_id, limit=32)
        st_items = [
            MemoryItem(
                id=it["id"],
                memory=it["text"],
                hash=None,
                created_at=it["created"],
                updated_at=None,
                score=float(0.95),  # ✅ FIXED: Reasonable STM score
            ).model_dump()
            for it in st_buf[-5:]
        ]

        merged = sorted(lt_items + st_items, key=lambda x: x["score"], reverse=True)
        return {"results": merged[:limit]}

    async def _create_memory(self, data: str, vec, meta: Dict) -> str:
        """Create memory with async support"""
        mid = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update(
            {
                "data": data,
                "hash": hashlib.md5(data.encode()).hexdigest(),
                "created_at": now,
                "__vector": vec,
            }
        )
        await asyncio.to_thread(self.vector_store.insert, [vec], [meta], [mid])
        await asyncio.to_thread(
            self.db.add_history, mid, None, data, "ADD", created_at=now
        )
        return mid

    async def _update_memory(self, mid, new_data, vec, meta):
        """Update memory with async support"""
        existing = await asyncio.to_thread(self.vector_store.search, "", vec, 1, {})
        created = existing[0].payload.get("created_at") if existing else None
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update(
            {
                "data": new_data,
                "hash": hashlib.md5(new_data.encode()).hexdigest(),
                "created_at": created,
                "updated_at": now,
                "__vector": vec,
            }
        )
        await asyncio.to_thread(self.vector_store.update, mid, vec, meta)
        await asyncio.to_thread(
            self.db.add_history,
            mid,
            existing[0].payload["data"] if existing else None,
            new_data,
            "UPDATE",
            created_at=created,
            updated_at=now,
        )

    async def _delete_memory(self, mid):
        """Delete memory with async support"""
        await asyncio.to_thread(self.vector_store.delete, mid)
        await asyncio.to_thread(
            self.db.add_history, mid, None, None, "DELETE", is_deleted=1
        )

    async def add_procedural_memory(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        prompt: Optional[str] = None,
    ) -> Dict:
        """Add procedural memory with proper async handling"""

        logger.info(f"Generating procedural memory for user_id={user_id}")

        prompt_msgs = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            # FIXED: Proper async call
            summary = await self.llm.generate_response_async(prompt_msgs)
        except Exception as e:
            logger.error(f"Failed to summarize procedural memory: {e}")
            raise

        # Store procedural memory
        vec = await asyncio.to_thread(self.embedder.embed, summary, "add")
        mid = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        metadata = {
            "user_id": user_id,
            "data": summary,
            "hash": hashlib.md5(summary.encode()).hexdigest(),
            "created_at": now,
            "memory_type": "procedural",
        }

        await asyncio.to_thread(self.vector_store.insert, [vec], [metadata], [mid])
        await asyncio.to_thread(
            self.db.add_history, mid, None, summary, "ADD", created_at=now
        )

        return {"results": [{"id": mid, "memory": summary, "event": "ADD"}]}

    async def get_all(self, user_id: str) -> Dict:
        """Get all memories for user"""
        filters = {"user_id": user_id}
        memories = await asyncio.to_thread(
            self.vector_store.list, filters=filters, limit=100
        )

        results = [
            {
                "id": mem.id,
                "memory": mem.payload.get("data", ""),
                "metadata": {
                    k: v
                    for k, v in mem.payload.items()
                    if k not in {"data", "__vector"}
                },
            }
            for mem in memories[0]
        ]
        return {"results": results}

    async def delete_all(self, user_id: str) -> None:
        """Delete all memories for user"""
        await asyncio.to_thread(self.vector_store.delete_col)
        self.short_term.clear(user_id)

    async def reset(self):
        """Reset memory system"""
        if hasattr(self.vector_store, "reset"):
            await asyncio.to_thread(self.vector_store.reset)
        else:
            raise NotImplementedError(
                "The current vector store does not support reset()."
            )
