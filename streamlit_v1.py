#!/usr/bin/env python3
"""
FRIDAY AI Assistant - Complete Streamlit Application
A production-ready chat interface with memory, RAG, and user management

Deploy with: streamlit run friday_streamlit_app.py
"""

import streamlit as st
import asyncio
import uuid
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 📱 STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FRIDAY AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-username/friday-assistant",
        "Report a bug": "https://github.com/your-username/friday-assistant/issues",
        "About": """
        # FRIDAY AI Assistant
        
        Your personal AI assistant with persistent memory, document processing, 
        and intelligent conversation capabilities.
        
        **Features:**
        - 🧠 Persistent Memory System
        - 📄 Document Processing & Search
        - 🤖 Advanced RAG Pipeline
        - 💬 Real-time Chat Interface
        - 👤 Multi-user Support
        
        Built with Python, Streamlit, and cutting-edge AI.
        """,
    },
)

# ============================================================================
# 🔧 CONFIGURATION & INITIALIZATION
# ============================================================================

# Set OpenAI API key from Streamlit secrets
try:
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
    elif not os.getenv("OPENAI_API_KEY"):
        st.error(
            "🚨 OpenAI API key not configured! Please set it in Streamlit secrets or environment variables."
        )
        st.stop()
except Exception as e:
    if not os.getenv("OPENAI_API_KEY"):
        st.error(f"🚨 Configuration error: {e}")
        st.stop()

# Data directories
DATA_DIR = "user_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# ============================================================================
# 💾 DATA PERSISTENCE FUNCTIONS
# ============================================================================


def load_users() -> List[str]:
    """Load user list from file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return []


def save_users(users: List[str]) -> None:
    """Save user list to file"""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving users: {e}")


def load_chat_history(user_id: str) -> List[Dict[str, Any]]:
    """Load chat history for a user"""
    try:
        history_file = os.path.join(CHAT_HISTORY_DIR, f"{user_id}.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading chat history for {user_id}: {e}")
        return []


def save_chat_history(user_id: str, history: List[Dict[str, Any]]) -> None:
    """Save chat history for a user"""
    try:
        history_file = os.path.join(CHAT_HISTORY_DIR, f"{user_id}.json")
        with open(history_file, "w") as f:
            json.dump(history[-100:], f, indent=2)  # Keep last 100 messages
    except Exception as e:
        logger.error(f"Error saving chat history for {user_id}: {e}")


# ============================================================================
# 🧠 MEMORY SYSTEM INITIALIZATION
# ============================================================================


@st.cache_resource
def initialize_memory_system():
    """Initialize the FRIDAY memory system (cached for performance)"""
    try:
        from my_mem.client import AsyncMemoryClient
        from my_mem.configs.base import MemoryConfig, LlmConfig

        # Configure async memory client
        config = MemoryConfig(
            llm=LlmConfig(provider="openai_async", config={}),
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": ".faiss_streamlit",
                    "collection_name": "friday_assistant",
                    "embedding_model_dims": 1536,
                },
            },
        )

        memory_client = AsyncMemoryClient(
            config=config,
            top_k=7,
            ltm_threshold=0.75,
            procedural_every_n=10,  # Auto-summarize every 10 messages
            enable_auto_summary=True,
        )

        logger.info("✅ FRIDAY Memory System initialized successfully")
        return memory_client

    except Exception as e:
        logger.error(f"❌ Failed to initialize memory system: {e}")
        st.error(f"Failed to initialize FRIDAY memory system: {e}")
        return None


# ============================================================================
# 🎨 CUSTOM CSS STYLING
# ============================================================================


def apply_custom_css():
    """Apply custom CSS for better UI"""
    st.markdown(
        """
    <style>
    /* Main container styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        position: relative;
    }
    
    .assistant-message {
        background: #f1f3f4;
        color: #333;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #4285f4;
    }
    
    /* Status indicators */
    .status-online { color: #34a853; }
    .status-thinking { color: #fbbc04; }
    .status-error { color: #ea4335; }
    
    /* Memory stats */
    .memory-stats {
        background: #e8f0fe;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar user card */
    .user-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 4px solid #4285f4;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 📊 SESSION STATE MANAGEMENT
# ============================================================================


def initialize_session_state():
    """Initialize all session state variables"""
    if "users" not in st.session_state:
        st.session_state.users = load_users()
        if not st.session_state.users:
            default_user = f"User-{uuid.uuid4().hex[:6]}"
            st.session_state.users = [default_user]
            save_users(st.session_state.users)

    if "selected_user" not in st.session_state:
        st.session_state.selected_user = (
            st.session_state.users[0] if st.session_state.users else None
        )

    if "memory_client" not in st.session_state:
        st.session_state.memory_client = initialize_memory_system()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "memory_stats" not in st.session_state:
        st.session_state.memory_stats = {}

    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True

    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True


# ============================================================================
# 👤 USER MANAGEMENT FUNCTIONS
# ============================================================================


def create_new_user(name_hint: str = "") -> str:
    """Create a new user with optional name hint"""
    if name_hint.strip():
        new_user = f"{name_hint.strip()}-{uuid.uuid4().hex[:4]}"
    else:
        new_user = f"User-{uuid.uuid4().hex[:6]}"

    st.session_state.users.append(new_user)
    save_users(st.session_state.users)
    return new_user


def delete_user(user_id: str) -> bool:
    """Delete a user and their data"""
    try:
        if user_id in st.session_state.users:
            st.session_state.users.remove(user_id)
            save_users(st.session_state.users)

            # Clean up user data
            history_file = os.path.join(CHAT_HISTORY_DIR, f"{user_id}.json")
            if os.path.exists(history_file):
                os.remove(history_file)

            # Clear from session state
            if user_id in st.session_state.chat_history:
                del st.session_state.chat_history[user_id]

            return True
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
    return False


# ============================================================================
# 🧠 MEMORY & CHAT FUNCTIONS
# ============================================================================


async def get_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get memory statistics for a user"""
    try:
        if st.session_state.memory_client:
            all_memories = await st.session_state.memory_client.get_all_memories(
                user_id=user_id
            )
            memories = all_memories.get("results", [])

            # Categorize memories
            categories = {}
            for mem in memories:
                mem_type = mem.get("metadata", {}).get("memory_type", "general")
                categories[mem_type] = categories.get(mem_type, 0) + 1

            return {
                "total": len(memories),
                "categories": categories,
                "recent": len([m for m in memories[-10:]]) if memories else 0,
            }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")

    return {"total": 0, "categories": {}, "recent": 0}


async def process_user_message(user_id: str, message: str) -> str:
    """Process user message through FRIDAY system"""
    try:
        if not st.session_state.memory_client:
            return "❌ Memory system not available. Please refresh the page."

        if st.session_state.streaming_enabled:
            # For streaming, we'll collect the full response
            full_response = ""
            async for token in st.session_state.memory_client.stream_rag(
                message, user_id=user_id
            ):
                full_response += token
            return full_response.strip()
        else:
            # Non-streaming response
            result = await st.session_state.memory_client.query_rag(
                message, user_id=user_id
            )
            return result.get("answer", "I couldn't generate a response.")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"❌ Error processing your message: {str(e)}"


async def stream_response(user_id: str, message: str):
    """Stream response token by token"""
    try:
        if st.session_state.memory_client:
            async for token in st.session_state.memory_client.stream_rag(
                message, user_id=user_id
            ):
                yield token
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield f"❌ Error: {str(e)}"


# ============================================================================
# 🎯 SIDEBAR - USER MANAGEMENT & CONTROLS
# ============================================================================


def render_sidebar():
    """Render the sidebar with user management and controls"""
    with st.sidebar:
        st.markdown("# 🤖 FRIDAY Assistant")
        st.markdown("---")

        # User Management Section
        st.markdown("### 👤 User Management")

        # Current user selection
        if st.session_state.users:
            selected_index = (
                st.session_state.users.index(st.session_state.selected_user)
                if st.session_state.selected_user in st.session_state.users
                else 0
            )

            new_selection = st.selectbox(
                "Active User",
                st.session_state.users,
                index=selected_index,
                help="Select the active user for the conversation",
            )

            # Handle user switch
            if new_selection != st.session_state.selected_user:
                st.session_state.selected_user = new_selection
                # Load chat history for new user
                if new_selection not in st.session_state.chat_history:
                    st.session_state.chat_history[new_selection] = load_chat_history(
                        new_selection
                    )
                st.rerun()

        # User creation
        with st.expander("➕ Add New User"):
            name_hint = st.text_input(
                "User Name (optional)", placeholder="e.g., Alice, Bob"
            )
            if st.button("Create User", type="primary"):
                new_user = create_new_user(name_hint)
                st.session_state.selected_user = new_user
                st.session_state.chat_history[new_user] = []
                st.success(f"✅ Created user: {new_user}")
                st.rerun()

        # Current user info
        if st.session_state.selected_user:
            st.markdown("#### 📊 Current User Stats")

            # Get memory stats asynchronously
            if st.session_state.selected_user not in st.session_state.memory_stats:
                with st.spinner("Loading memory stats..."):
                    stats = asyncio.run(
                        get_memory_stats(st.session_state.selected_user)
                    )
                    st.session_state.memory_stats[st.session_state.selected_user] = (
                        stats
                    )

            stats = st.session_state.memory_stats.get(
                st.session_state.selected_user, {}
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("💭 Memories", stats.get("total", 0))
            with col2:
                st.metric("📚 Recent", stats.get("recent", 0))

            # Memory categories
            categories = stats.get("categories", {})
            if categories:
                st.markdown("**Memory Types:**")
                for cat, count in categories.items():
                    st.markdown(f"• {cat}: {count}")

        st.markdown("---")

        # Settings Section
        st.markdown("### ⚙️ Settings")

        st.session_state.streaming_enabled = st.checkbox(
            "🌊 Enable Streaming",
            value=st.session_state.streaming_enabled,
            help="Stream responses in real-time",
        )

        st.session_state.auto_save = st.checkbox(
            "💾 Auto-save Chat",
            value=st.session_state.auto_save,
            help="Automatically save chat history",
        )

        # Advanced actions
        with st.expander("🔧 Advanced Actions"):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🧹 Clear Chat", help="Clear current chat history"):
                    if st.session_state.selected_user:
                        st.session_state.chat_history[
                            st.session_state.selected_user
                        ] = []
                        st.success("Chat cleared!")
                        st.rerun()

            with col2:
                if st.button("� Refresh Stats", help="Refresh memory statistics"):
                    if st.session_state.selected_user:
                        stats = asyncio.run(
                            get_memory_stats(st.session_state.selected_user)
                        )
                        st.session_state.memory_stats[
                            st.session_state.selected_user
                        ] = stats
                        st.success("Stats refreshed!")
                        st.rerun()

            if st.button(
                "🗑️ Delete Current User", help="Permanently delete current user"
            ):
                if st.session_state.selected_user and len(st.session_state.users) > 1:
                    if delete_user(st.session_state.selected_user):
                        st.session_state.selected_user = st.session_state.users[0]
                        st.success("User deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete user")
                else:
                    st.warning("Cannot delete the only user")

        st.markdown("---")

        # System Status
        st.markdown("### 📡 System Status")

        status_color = "🟢" if st.session_state.memory_client else "🔴"
        status_text = "Online" if st.session_state.memory_client else "Offline"
        st.markdown(f"{status_color} **Memory System:** {status_text}")

        if st.session_state.processing:
            st.markdown("� **Status:** Processing...")
        else:
            st.markdown("� **Status:** Ready")

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8em;'>
                <p>🤖 FRIDAY AI Assistant</p>
                <p>Built with ❤️ and AI</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# 💬 MAIN CHAT INTERFACE
# ============================================================================


def render_chat_interface():
    """Render the main chat interface"""
    st.markdown("## 💬 Chat with FRIDAY")

    if not st.session_state.selected_user:
        st.warning(
            "👈 Please select or create a user in the sidebar to start chatting."
        )
        return

    if not st.session_state.memory_client:
        st.error("❌ Memory system not available. Please check your configuration.")
        return

    # Load chat history for current user
    current_user = st.session_state.selected_user
    if current_user not in st.session_state.chat_history:
        st.session_state.chat_history[current_user] = load_chat_history(current_user)

    # Display chat history
    chat_container = st.container()

    with chat_container:
        # Welcome message for new users
        if not st.session_state.chat_history[current_user]:
            st.markdown(
                """
                <div class="assistant-message">
                    <p>👋 <strong>Hello! I'm FRIDAY, your personal AI assistant.</strong></p>
                    <p>I can help you with:</p>
                    <ul>
                        <li>🧠 Remembering information about you</li>
                        <li>💬 Having intelligent conversations</li>
                        <li>📚 Learning from our interactions</li>
                        <li>🎯 Providing personalized assistance</li>
                    </ul>
                    <p>Try asking me something or telling me about yourself!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display chat messages
        for i, message in enumerate(st.session_state.chat_history[current_user]):
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="user-message">
                        <strong>You:</strong><br>
                        {message["content"]}
                        <div style="font-size: 0.8em; opacity: 0.8; margin-top: 0.5rem;">
                            {message.get("timestamp", "")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="assistant-message">
                        <strong>🤖 FRIDAY:</strong><br>
                        {message["content"]}
                        <div style="font-size: 0.8em; opacity: 0.6; margin-top: 0.5rem;">
                            {message.get("timestamp", "")} • Processing: {message.get("processing_time", "N/A")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Chat input
    st.markdown("---")

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_area(
                "Message FRIDAY:",
                placeholder="Ask me anything or tell me about yourself...",
                height=100,
                key="user_input",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            submit_button = st.form_submit_button(
                "Send 📤", type="primary", use_container_width=True
            )

    # Process message when submitted
    if submit_button and user_input.strip():
        process_message(current_user, user_input.strip())


def process_message(user_id: str, message: str):
    """Process and display a user message"""
    # Add user message to history
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {"role": "user", "content": message, "timestamp": timestamp}

    st.session_state.chat_history[user_id].append(user_message)
    st.session_state.processing = True

    # Display user message immediately
    st.markdown(
        f"""
        <div class="user-message">
            <strong>You:</strong><br>
            {message}
            <div style="font-size: 0.8em; opacity: 0.8; margin-top: 0.5rem;">
                {timestamp}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Process response
    response_placeholder = st.empty()

    with response_placeholder:
        if st.session_state.streaming_enabled:
            # Streaming response
            with st.spinner("🤖 FRIDAY is thinking..."):
                start_time = time.time()

                # Create response container
                response_container = st.empty()
                full_response = ""

                try:
                    # Stream the response
                    async def stream_and_display():
                        nonlocal full_response
                        async for token in stream_response(user_id, message):
                            full_response += token
                            response_container.markdown(
                                f"""
                                <div class="assistant-message">
                                    <strong>🤖 FRIDAY:</strong><br>
                                    {full_response}<span style="opacity: 0.5;">▌</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Run streaming
                    asyncio.run(stream_and_display())

                    processing_time = f"{time.time() - start_time:.1f}s"

                    # Final response without cursor
                    response_container.markdown(
                        f"""
                        <div class="assistant-message">
                            <strong>🤖 FRIDAY:</strong><br>
                            {full_response}
                            <div style="font-size: 0.8em; opacity: 0.6; margin-top: 0.5rem;">
                                {timestamp} • Processing: {processing_time}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    full_response = f"❌ Sorry, I encountered an error: {str(e)}"
                    processing_time = f"{time.time() - start_time:.1f}s"
        else:
            # Non-streaming response
            with st.spinner("🤖 FRIDAY is thinking..."):
                start_time = time.time()
                full_response = asyncio.run(process_user_message(user_id, message))
                processing_time = f"{time.time() - start_time:.1f}s"

    # Add assistant response to history
    assistant_message = {
        "role": "assistant",
        "content": full_response,
        "timestamp": timestamp,
        "processing_time": processing_time,
    }

    st.session_state.chat_history[user_id].append(assistant_message)
    st.session_state.processing = False

    # Auto-save chat history
    if st.session_state.auto_save:
        save_chat_history(user_id, st.session_state.chat_history[user_id])

    # Refresh memory stats
    if user_id in st.session_state.memory_stats:
        del st.session_state.memory_stats[user_id]  # Will be reloaded

    # Rerun to update the interface
    st.rerun()


# ============================================================================
# 🎯 MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point"""
    # Apply custom styling
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Check if memory system is available
    if not st.session_state.memory_client:
        st.error(
            """
            ❌ **FRIDAY Memory System Not Available**
            
            Please ensure:
            1. OpenAI API key is properly configured
            2. All dependencies are installed
            3. Network connection is available
            
            Check the logs for more details.
            """
        )
        return

    # Render sidebar
    render_sidebar()

    # Main content area
    if st.session_state.selected_user:
        render_chat_interface()
    else:
        st.markdown(
            """
            # 🤖 Welcome to FRIDAY AI Assistant
            
            Your personal AI assistant with advanced memory capabilities.
            
            ## 🚀 Getting Started
            
            1. **Create a user** in the sidebar to begin
            2. **Start chatting** with FRIDAY about anything
            3. **Build your memory** as FRIDAY learns about you
            4. **Enjoy personalized** AI assistance
            
            ## ✨ Features
            
            - 🧠 **Persistent Memory**: FRIDAY remembers our conversations
            - 💬 **Intelligent Chat**: Context-aware responses  
            - 🎯 **Personalized**: Learns your preferences and habits
            - 📱 **Multi-User**: Support for multiple user profiles
            - 🌊 **Real-time**: Streaming responses like ChatGPT
            
            **👈 Start by creating a user in the sidebar!**
            """
        )


# ============================================================================
# 🏃‍♂️ APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
