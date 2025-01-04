import streamlit as st
from v2 import SubtitleChatbot
import os
from datetime import datetime
import json
import gtts
from io import BytesIO
import time
import pyperclip

# Set page configuration
st.set_page_config(
    page_title="YouTube Assistant",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = {"id": int(time.time()), "title": "New Chat", "messages": []}

class FeedbackStore:
    def __init__(self, filename="feedback_data.json"):
        self.filename = filename
        self.feedback_data = self.load_feedback()

    def load_feedback(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_feedback(self, question, answer, is_helpful, improvement_feedback=None):
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'is_helpful': is_helpful,
            'improvement_feedback': improvement_feedback
        }
        self.feedback_data.append(feedback_entry)
        
        with open(self.filename, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

def initialize_chatbot():
    """Initialize the chatbot and load the model"""
    chatbot = SubtitleChatbot()
    
    if os.path.exists("subtitle_vectorstore"):
        return chatbot.load_model()
    else:
        return chatbot.train()

def save_chat_history():
    """Save chat history to a JSON file"""
    try:
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.chat_history_list, f)
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")

def load_chat_history():
    """Load chat history from JSON file"""
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gtts.gTTS(text)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"Failed to convert text to speech: {str(e)}")
        return None

def copy_to_clipboard(text):
    """Copy text to clipboard and show success message"""
    try:
        pyperclip.copy(text)
        st.toast('Copied to clipboard!', icon='✂️')
    except Exception as e:
        st.error(f"Failed to copy: {str(e)}")

# Main app layout
st.title("🎥 YouTube Assistant")
st.markdown("---")

# Initialize chatbot if not already done
if st.session_state.chatbot is None:
    with st.spinner("Initializing chatbot... This may take a few minutes."):
        qa_chain = initialize_chatbot()
        if qa_chain:
            st.session_state.chatbot = qa_chain
            st.success("Chatbot initialized successfully!")
        else:
            st.error("Failed to initialize chatbot. Please check the logs.")

# Add sidebar with chat history
with st.sidebar:
    st.title("Chats")
    
    # New chat button
    if st.button("+ New Chat"):
        st.session_state.current_chat = {
            "id": int(time.time()),
            "title": "New Chat",
            "messages": []
        }
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    for chat in st.session_state.chat_history_list:
        if st.button(
            f"💬 {chat['title'][:30]}...",
            key=f"chat_{chat['id']}",
            use_container_width=True
        ):
            st.session_state.current_chat = chat
            st.session_state.messages = chat['messages']
            st.rerun()
    
    # About section at the bottom
    st.markdown("---")
    st.title("About")
    st.markdown("""
    This YouTube Assistant helps you explore Nikhil Kamath's content through an 
    AI-powered conversation. Ask questions about topics discussed in his videos!
    
    **Features:**
    - 💬 Natural conversation interface
    - 🎥 Source video links
    - 🔊 Text-to-speech
    - 💾 Chat history
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # First write the message content
        st.markdown(message["content"])
        
        # Only show action buttons for assistant messages
        if message["role"] == "assistant":
            # Add a small space between message and buttons
            st.markdown("")  # Empty line for spacing
            
            # Create columns for buttons with smaller width ratios
            cols = st.columns([0.5, 0.5, 0.5, 0.5, 8])  # Make buttons more compact
            
            # Audio button
            with cols[0]:
                if st.button("🔊", key=f"audio_{hash(message['content'])}", help="Read aloud"):
                    audio = text_to_speech(message["content"])
                    if audio:
                        st.audio(audio, format='audio/mp3')
            
            # Copy button
            with cols[1]:
                if st.button("📋", key=f"copy_{hash(message['content'])}", help="Copy to clipboard"):
                    copy_to_clipboard(message["content"])
            
            # Helpful button
            with cols[2]:
                if st.button("👍", key=f"helpful_{hash(message['content'])}", help="Mark as helpful"):
                    st.toast("Thanks for your feedback!", icon="👍")
            
            # Not helpful button
            with cols[3]:
                if st.button("👎", key=f"not_helpful_{hash(message['content'])}", help="Mark as not helpful"):
                    st.toast("Thanks for your feedback!", icon="👎")

            # Show sources if available
            if "sources" in message:
                st.markdown("---")
                st.markdown("**Source Videos:**")
                for source in message["sources"]:
                    if "video_url" in source:
                        st.markdown(f"📺 [Watch Video]({source['video_url']})")



# Chat input
if prompt := st.chat_input("Ask me anything about Nikhil Kamath..."):
    # Update current chat title if it's the first message
    if not st.session_state.current_chat["messages"]:
        st.session_state.current_chat["title"] = prompt[:50]
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.current_chat["messages"] = st.session_state.messages
    
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.chatbot:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chatbot({"question": prompt})
                    response = result["answer"]
                    st.write(response)
                    
                    # Add text-to-speech for response
                    audio = text_to_speech(response)
                    if audio:
                        st.audio(audio, format='audio/mp3')

                    # Process sources
                    unique_urls = set()
                    if "source_documents" in result:
                        for doc in result["source_documents"]:
                            if 'video_url' in doc.metadata:
                                unique_urls.add(doc.metadata['video_url'])
                    
                    if unique_urls:
                        st.markdown("---")
                        st.markdown("**Source Videos:**")
                        for url in unique_urls:
                            st.markdown(f"📺 [Watch Video]({url})")

                    # Update messages and save chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": [{"video_url": url} for url in unique_urls]
                    })
                    st.session_state.current_chat["messages"] = st.session_state.messages
                    
                    # Update chat history
                    existing_chat = next(
                        (chat for chat in st.session_state.chat_history_list 
                         if chat["id"] == st.session_state.current_chat["id"]), 
                        None
                    )
                    if existing_chat:
                        existing_chat.update(st.session_state.current_chat)
                    else:
                        st.session_state.chat_history_list.append(st.session_state.current_chat)
                    
                    save_chat_history()

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
    else:
        st.error("Chatbot is not initialized. Please refresh the page.")
        