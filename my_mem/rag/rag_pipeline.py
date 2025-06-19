# Updated my_mem/rag/rag_pipeline.py - Replace the AsyncRAGPipeline class

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Generator, AsyncGenerator

from my_mem.memory.main import Memory, AsyncMemory
from my_mem.utils.factory import LlmFactory
from my_mem.configs.base import MemoryConfig

logger = logging.getLogger(__name__)


_CITATION_SYSTEM_PROMPT = """
You are a helpful assistant engaged in an ongoing conversation with a user.
You have access to the user's short-term memory (recent turns) and long-term memory (facts, history, preferences).

Use this context to answer **naturally and conversationally**.
If a follow-up depends on recent turns, maintain continuity.
If useful, leverage known facts about the user — but avoid repeating them unless relevant.

Always prefer **fluid, friendly, and accurate responses**.
Never mention memory IDs or numbered sources.

If you don't have enough information, it's okay to ask clarifying questions.
"""


def _build_context(results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Turns Memory.search() results → numbered context block.
    Returns the block **and** a light "sources" list for downstream use.
    """
    lines = []
    sources = []
    for idx, hit in enumerate(results, start=1):
        label = f"[{idx}]"
        text = hit["memory"]
        lines.append(f"{label} {text}")
        sources.append({"id": hit["id"], "text": text})
    return "\n".join(lines), sources


class RAGPipeline:
    """Small wrapper so you can do:  rag = RAGPipeline(mem); rag.query("…")"""

    def __init__(self, memory: Memory, top_k: int = 5, ltm_threshold: float = 0.75):
        self.memory = memory
        self.top_k = top_k
        self.ltm_threshold = ltm_threshold
        self.llm = memory.llm

    def query(self, question: str, *, user_id: str) -> Dict[str, Any]:
        """Run RAG & return { answer, sources }."""
        retrieved = self.memory.search(
            question,
            user_id=user_id,
            limit=self.top_k,
            ltm_threshold=self.ltm_threshold,
        )["results"]
        logger.debug(f"RAG retrieved {len(retrieved)} memories")

        context_block, sources = _build_context(retrieved)
        answer = self._ask_llm(question, context_block)

        return {"answer": answer, "sources": sources}

    def stream_query(self, query: str, user_id: str) -> Generator[str, None, None]:
        retrieved = self.memory.search(
            query, user_id=user_id, limit=self.top_k, ltm_threshold=self.ltm_threshold
        )["results"]
        logger.debug(f"RAG retrieved {len(retrieved)} memories")

        context_block, _ = _build_context(retrieved)
        system_prompt = _CITATION_SYSTEM_PROMPT
        user_prompt = f"Context:\n{context_block}\n\nQ: {query}"

        yield from self.llm.stream_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def _ask_llm(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": _CITATION_SYSTEM_PROMPT},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": question},
        ]
        resp = self.llm.generate_response(messages=messages)
        return resp.strip()


class AsyncRAGPipeline:
    def __init__(
        self, memory: AsyncMemory, top_k: int = 5, ltm_threshold: float = 0.75
    ):
        self.memory = memory
        self.top_k = top_k
        self.ltm_threshold = ltm_threshold
        self.llm = memory.llm

    async def query(self, query: str, *, user_id: str) -> Dict[str, Any]:
        """FIXED: Proper async RAG query"""
        results = (
            await self.memory.search(
                query,
                user_id=user_id,
                limit=self.top_k,
                ltm_threshold=self.ltm_threshold,
            )
        )["results"]
        logger.debug(f"RAG retrieved {len(results)} memories")

        context_block, sources = _build_context(results)
        answer = await self._ask_llm(query, context_block)
        return {"answer": answer, "sources": sources}

    async def stream_query(
        self, query: str, *, user_id: str
    ) -> AsyncGenerator[str, None, None]:
        """FIXED: Proper async streaming"""
        results = (
            await self.memory.search(
                query,
                user_id=user_id,
                limit=self.top_k,
                ltm_threshold=self.ltm_threshold,
            )
        )["results"]

        context_block, _ = _build_context(results)
        system_msg = _CITATION_SYSTEM_PROMPT
        user_msg = f"Context:\n{context_block}\n\nQ: {query}"

        # FIXED: Check if llm has async stream method
        if hasattr(self.llm, "stream_response_async"):
            async for chunk in self.llm.stream_response_async(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            ):
                yield chunk
        else:
            # Fallback: Generate complete response and yield in chunks
            answer = await self._ask_llm(query, context_block)
            # Simulate streaming by yielding words
            words = answer.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word

    async def _ask_llm(self, query: str, context: str) -> str:
        """FIXED: Proper async LLM call"""
        messages = [
            {"role": "system", "content": _CITATION_SYSTEM_PROMPT},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": query},
        ]

        # FIXED: Use proper async method
        try:
            response = await self.llm.generate_response_async(messages)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM async call failed: {e}")
            # Fallback: if async doesn't work, try sync
            if hasattr(self.llm, "generate_response"):
                response = self.llm.generate_response(messages=messages)
                return response.strip()
            else:
                raise e


def get_default_rag(top_k: int = 5) -> RAGPipeline:
    """Utility for rapid prototyping"""
    mem = Memory(MemoryConfig())
    return RAGPipeline(mem, top_k=top_k)
