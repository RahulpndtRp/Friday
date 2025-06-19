from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Optional
import json
import uuid
import asyncio

from src.core.chat.chat_manager import ChatManager
from src.core.telemetry.logger import StructuredLogger

# Create WebSocket router
router = APIRouter()
logger = StructuredLogger("websocket.chat")


class WebSocketManager:
    """Manages WebSocket connections for real-time chat."""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, any]] = {}

    async def connect(
        self, websocket: WebSocket, user_id: str, chat_manager: ChatManager
    ):
        """Accept WebSocket connection."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())

        self.active_connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "chat_manager": chat_manager,
        }

        logger.info(
            f"WebSocket connected", user_id=user_id, connection_id=connection_id
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection",
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "message": "Connected to FRIDAY Assistant",
                }
            )
        )

        return connection_id

    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            user_id = self.active_connections[connection_id]["user_id"]
            del self.active_connections[connection_id]
            logger.info(
                f"WebSocket disconnected", connection_id=connection_id, user_id=user_id
            )

    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection."""
        connection = self.active_connections.get(connection_id)
        if connection:
            try:
                await connection["websocket"].send_text(json.dumps(message))
            except:
                self.disconnect(connection_id)

    def get_connection(self, connection_id: str) -> Optional[Dict[str, any]]:
        """Get connection details."""
        return self.active_connections.get(connection_id)


# Global WebSocket manager
websocket_manager = WebSocketManager()

# Dependency injection for chat managers
_chat_managers: Dict[str, ChatManager] = {}


def set_chat_manager(user_id: str, chat_manager: ChatManager):
    """Set chat manager for WebSocket use."""
    _chat_managers[user_id] = chat_manager


@router.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat."""
    # Get chat manager for user
    chat_manager = _chat_managers.get(user_id)
    if not chat_manager:
        await websocket.close(code=1003, reason="Chat service not available")
        return

    connection_id = await websocket_manager.connect(websocket, user_id, chat_manager)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                await handle_websocket_message(connection_id, message_data)

            except json.JSONDecodeError:
                await websocket_manager.send_message(
                    connection_id, {"type": "error", "message": "Invalid JSON format"}
                )
            except Exception as e:
                await websocket_manager.send_message(
                    connection_id, {"type": "error", "message": str(e)}
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error", connection_id=connection_id, error=str(e))
        websocket_manager.disconnect(connection_id)


async def handle_websocket_message(connection_id: str, message_data: dict):
    """Handle incoming WebSocket message."""
    connection = websocket_manager.get_connection(connection_id)
    if not connection:
        return

    chat_manager = connection["chat_manager"]
    message_type = message_data.get("type")

    if message_type == "chat":
        # Handle chat message
        user_message = message_data.get("message", "")
        conversation_id = message_data.get("conversation_id")

        if not user_message.strip():
            await websocket_manager.send_message(
                connection_id, {"type": "error", "message": "Empty message not allowed"}
            )
            return

        try:
            # Send streaming response
            stream = await chat_manager.send_message(
                message=user_message, conversation_id=conversation_id, stream=True
            )

            async for chunk in stream:
                await websocket_manager.send_message(
                    connection_id,
                    {
                        "type": "stream_chunk",
                        "chunk": {
                            "chunk_id": chunk.chunk_id,
                            "message_id": chunk.message_id,
                            "content": chunk.content,
                            "is_final": chunk.is_final,
                            "metadata": chunk.metadata,
                        },
                    },
                )

        except Exception as e:
            await websocket_manager.send_message(
                connection_id,
                {"type": "error", "message": f"Failed to process message: {str(e)}"},
            )

    elif message_type == "ping":
        # Handle ping/pong for connection keepalive
        await websocket_manager.send_message(
            connection_id, {"type": "pong", "timestamp": message_data.get("timestamp")}
        )

    else:
        await websocket_manager.send_message(
            connection_id,
            {"type": "error", "message": f"Unknown message type: {message_type}"},
        )
