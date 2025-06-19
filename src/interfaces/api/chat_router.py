from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
import json
import asyncio

from src.core.chat.models import ChatMessageRequest, ChatMessageResponse, StreamChunk
from src.core.chat.chat_manager import ChatManager
from src.core.telemetry.logger import StructuredLogger

# Create router
router = APIRouter(prefix="/chat", tags=["chat"])
logger = StructuredLogger("api.chat")

# Dependency injection - will be set by main app
_chat_managers: Dict[str, ChatManager] = {}


def set_chat_manager(user_id: str, chat_manager: ChatManager):
    """Set chat manager for dependency injection."""
    _chat_managers[user_id] = chat_manager


async def get_chat_manager(
    user_id: Optional[str] = Query(default="friday_user_001"),
) -> ChatManager:
    """Dependency to get chat manager for user."""
    chat_manager = _chat_managers.get(user_id)
    if not chat_manager:
        # Try default user if requested user not found
        chat_manager = _chat_managers.get("friday_user_001")
        if not chat_manager:
            available_users = list(_chat_managers.keys())
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Chat service not available for user {user_id}. Available users: {available_users}",
            )
    return chat_manager


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    user_id: Optional[str] = Query(default="friday_user_001", description="User ID"),
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    """Send a chat message and get response."""
    try:
        logger.info(
            f"Received chat message",
            user_id=user_id,
            message_length=len(request.message),
            stream=request.stream,
        )

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_chat_response(chat_manager, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"},
            )
        else:
            # Return complete response
            response = await chat_manager.send_message(
                message=request.message,
                conversation_id=request.conversation_id,
                stream=False,
            )

            logger.info(
                f"Chat response generated",
                user_id=user_id,
                message_id=response.message_id,
                processing_time=response.processing_time,
            )

            return response

    except Exception as e:
        logger.error(f"Failed to process chat message", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def stream_chat_response(chat_manager: ChatManager, request: ChatMessageRequest):
    """Stream chat response chunks."""
    try:
        stream = await chat_manager.send_message(
            message=request.message,
            conversation_id=request.conversation_id,
            stream=True,
        )

        async for chunk in stream:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "message_id": chunk.message_id,
                "content": chunk.content,
                "is_final": chunk.is_final,
                "metadata": chunk.metadata,
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

            # Add small delay to prevent overwhelming
            await asyncio.sleep(0.01)

    except Exception as e:
        error_chunk = {
            "chunk_id": "error",
            "message_id": "error",
            "content": f"Error: {str(e)}",
            "is_final": True,
            "metadata": {"error": True},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.get("/conversations")
async def get_conversations(
    limit: Optional[int] = 20,
    user_id: Optional[str] = Query(default="friday_user_001", description="User ID"),
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    """Get user's conversations."""
    try:
        conversations = await chat_manager.get_user_conversations(limit)
        return {
            "conversations": [conv.to_dict() for conv in conversations],
            "count": len(conversations),
            "user_id": user_id,
        }
    except Exception as e:
        logger.error(f"Failed to get conversations", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user_id: Optional[str] = Query(default="friday_user_001", description="User ID"),
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    """Get conversation details."""
    try:
        conversation = await chat_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        return conversation.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get conversation", conversation_id=conversation_id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: Optional[int] = 50,
    user_id: Optional[str] = Query(default="friday_user_001", description="User ID"),
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    """Get messages for a conversation."""
    try:
        messages = await chat_manager.get_conversation_messages(conversation_id, limit)
        return {
            "messages": [msg.to_dict() for msg in messages],
            "count": len(messages),
            "conversation_id": conversation_id,
        }
    except Exception as e:
        logger.error(
            f"Failed to get conversation messages",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/search")
async def search_conversations(
    q: str,
    limit: Optional[int] = 10,
    user_id: Optional[str] = Query(default="friday_user_001", description="User ID"),
    chat_manager: ChatManager = Depends(get_chat_manager),
):
    """Search across conversations."""
    try:
        messages = await chat_manager.search_conversations(q, limit)
        return {
            "results": [msg.to_dict() for msg in messages],
            "count": len(messages),
            "query": q,
        }
    except Exception as e:
        logger.error(f"Failed to search conversations", query=q, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
