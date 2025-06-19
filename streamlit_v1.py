#!/usr/bin/env python3
"""
FRIDAY AI Assistant - Enhanced Streamlit Application
Advanced UI with better chat experience, memory categorization, and modern design

Deploy with: streamlit run enhanced_friday_streamlit.py
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
        # FRIDAY AI Assistant v2.0
        
        Your advanced personal AI assistant with persistent memory, intelligent conversation,
        and real-time learning capabilities.
        
        **Enhanced Features:**
        - 🧠 Advanced Memory Categorization
        - 📄 Smart Context Understanding
        - 🤖 Improved RAG Pipeline
        - 💬 Modern Chat Interface
        - 👤 Multi-user Support
        - 🎨 Beautiful UI/UX
        
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
# 🎨 ENHANCED CSS STYLING
# ============================================================================


def apply_enhanced_css():
    """Apply enhanced CSS for modern UI"""
    st.markdown(
        """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stCheckbox label,
    .css-1d391kg .stTextInput label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Enhanced Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 15%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        position: relative;
        animation: slideInRight 0.3s ease-out;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        color: #2c3e50;
        padding: 1.2rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 15% 1rem 0;
        border-left: 4px solid #4285f4;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        position: relative;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Animation keyframes */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Typing indicator */
    .typing-indicator {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 20px;
        margin: 1rem 15% 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        animation: pulse 1.5s infinite;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #4285f4;
        border-radius: 50%;
        animation: bounce 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Enhanced Status Indicators */
    .status-online { 
        color: #34a853; 
        font-weight: 600;
        text-shadow: 0 0 5px rgba(52, 168, 83, 0.3);
    }
    .status-thinking { 
        color: #fbbc04; 
        font-weight: 600;
        text-shadow: 0 0 5px rgba(251, 188, 4, 0.3);
    }
    .status-error { 
        color: #ea4335; 
        font-weight: 600;
        text-shadow: 0 0 5px rgba(234, 67, 53, 0.3);
    }
    
    /* Enhanced Memory Stats */
    .memory-stats {
        background: linear-gradient(135deg, #e8f0fe 0%, #f8f9ff 100%);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid #e1e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .memory-category {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe8cc 100%);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #ff9800;
        font-size: 0.9rem;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Send button special styling */
    .send-button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 15px !important;
    }
    
    /* Enhanced Input Field */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e1e8f0;
        padding: 1rem;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        resize: none;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #4285f4;
        box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1);
        outline: none;
    }
    
    /* User card in sidebar */
    .user-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4285f4;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Welcome message styling */
    .welcome-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 2px dashed #4285f4;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chat container */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f3f4;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #4285f4;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #3367d6;
    }
    
    /* Input container */
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        border-top: 1px solid #e1e8f0;
        padding: 1.5rem;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Message timestamp */
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        text-align: right;
    }
    
    /* Processing indicator */
    .processing-indicator {
        font-size: 0.8rem;
        color: #4285f4;
        font-weight: 500;
    }
    
    /* Custom selectbox */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Memory type badges */
    .memory-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.1rem;
    }
    
    .badge-personal { background: #e3f2fd; color: #1976d2; }
    .badge-work { background: #f3e5f5; color: #7b1fa2; }
    .badge-interests { background: #e8f5e8; color: #388e3c; }
    .badge-general { background: #fff3e0; color: #f57c00; }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 💾 DATA PERSISTENCE FUNCTIONS (Enhanced)
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
            json.dump(history[-200:], f, indent=2)  # Keep last 200 messages
    except Exception as e:
        logger.error(f"Error saving chat history for {user_id}: {e}")


# ============================================================================
# 🧠 ENHANCED MEMORY SYSTEM
# ============================================================================


@st.cache_resource
def initialize_memory_system():
    """Initialize the FRIDAY memory system with enhanced configuration"""
    try:
        from my_mem.client import AsyncMemoryClient
        from my_mem.configs.base import MemoryConfig, LlmConfig

        # Enhanced configuration
        config = MemoryConfig(
            llm=LlmConfig(provider="openai_async", config={}),
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": ".faiss_streamlit_enhanced",
                    "collection_name": "friday_assistant_v2",
                    "embedding_model_dims": 1536,
                },
            },
        )

        memory_client = AsyncMemoryClient(
            config=config,
            top_k=8,  # More context
            ltm_threshold=0.70,  # Lower threshold for more recall
            procedural_every_n=8,  # More frequent summarization
            enable_auto_summary=True,
        )

        logger.info("✅ Enhanced FRIDAY Memory System initialized")
        return memory_client

    except Exception as e:
        logger.error(f"❌ Failed to initialize memory system: {e}")
        st.error(f"Failed to initialize FRIDAY memory system: {e}")
        return None


# ============================================================================
# 📊 ENHANCED MEMORY ANALYTICS
# ============================================================================


async def get_enhanced_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get enhanced memory statistics with categorization"""
    try:
        if st.session_state.memory_client:
            all_memories = await st.session_state.memory_client.get_all_memories(
                user_id=user_id
            )
            memories = all_memories.get("results", [])

            # Enhanced categorization
            categories = {
                "personal": 0,
                "work": 0,
                "interests": 0,
                "procedural": 0,
                "general": 0,
            }

            recent_count = 0

            for mem in memories:
                content = mem.get("memory", "").lower()
                mem_type = mem.get("metadata", {}).get("memory_type", "general")

                # Smart categorization based on content
                if any(
                    word in content
                    for word in [
                        "i am",
                        "my name",
                        "i live",
                        "i work",
                        "i like",
                        "i love",
                        "i prefer",
                    ]
                ):
                    categories["personal"] += 1
                elif any(
                    word in content
                    for word in [
                        "work",
                        "job",
                        "career",
                        "office",
                        "project",
                        "meeting",
                    ]
                ):
                    categories["work"] += 1
                elif any(
                    word in content
                    for word in [
                        "book",
                        "movie",
                        "music",
                        "hobby",
                        "sport",
                        "game",
                        "read",
                    ]
                ):
                    categories["interests"] += 1
                elif mem_type == "procedural":
                    categories["procedural"] += 1
                else:
                    categories["general"] += 1

                # Count recent memories (last 24 hours)
                try:
                    created_at = mem.get("created_at", "")
                    if created_at:
                        # Simple check for recent memories
                        recent_count += (
                            1 if len(memories) - memories.index(mem) <= 10 else 0
                        )
                except:
                    pass

            return {
                "total": len(memories),
                "categories": categories,
                "recent": min(recent_count, 10),
                "active_categories": len([c for c in categories.values() if c > 0]),
            }
    except Exception as e:
        logger.error(f"Error getting enhanced memory stats: {e}")

    return {"total": 0, "categories": {}, "recent": 0, "active_categories": 0}


# ============================================================================
# 💬 ENHANCED CHAT PROCESSING
# ============================================================================


async def process_user_message_enhanced(user_id: str, message: str) -> str:
    """Enhanced message processing with better error handling"""
    try:
        if not st.session_state.memory_client:
            return "❌ Memory system not available. Please refresh the page."

        # Process through enhanced RAG
        result = await st.session_state.memory_client.query_rag(
            message, user_id=user_id
        )
        response = result.get("answer", "I couldn't generate a response.")

        # Add source information if available
        sources = result.get("sources", [])
        if sources and len(response) > 50:  # Only add sources for substantial responses
            source_count = len(sources)
            response += f"\n\n*💡 Response based on {source_count} memory{'ies' if source_count > 1 else 'y'}*"

        return response.strip()

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"❌ I encountered an error processing your message. Please try again."


async def stream_response_enhanced(user_id: str, message: str):
    """Enhanced streaming with better error handling"""
    try:
        if st.session_state.memory_client:
            async for token in st.session_state.memory_client.stream_rag(
                message, user_id=user_id
            ):
                yield token
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield f"❌ Sorry, I encountered an error while processing your message."


# ============================================================================
# 📊 SESSION STATE MANAGEMENT (Enhanced)
# ============================================================================


def initialize_session_state():
    """Initialize enhanced session state variables"""
    defaults = {
        "users": load_users() or [f"User-{uuid.uuid4().hex[:6]}"],
        "selected_user": None,
        "memory_client": initialize_memory_system(),
        "chat_history": {},
        "processing": False,
        "memory_stats": {},
        "show_settings": False,
        "auto_save": True,
        "streaming_enabled": True,
        "message_count": 0,
        "typing_indicator": False,
        "last_activity": time.time(),
        "chat_input_key": 0,  # For clearing input after send
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Set initial selected user
    if not st.session_state.selected_user and st.session_state.users:
        st.session_state.selected_user = st.session_state.users[0]

    # Save users if they were created
    if len(st.session_state.users) == 1 and not load_users():
        save_users(st.session_state.users)


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
