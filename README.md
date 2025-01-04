YouTube Assistant Chatbot
The YouTube Assistant Chatbot is an AI-powered conversational tool designed to help users explore and interact with the content of YouTube videos effectively. This bot has been created using Nikhil Kamath's YouTube channel as a reference to demonstrate its capabilities. The project integrates advanced language models like Gemini and document management techniques to deliver accurate, context-aware answers based on video subtitles.

<br>
Features
🎥 Interactive Video Exploration: <br> Ask questions about YouTube videos, and the chatbot provides concise and relevant answers based on subtitles. <br>

📄 Subtitle Management: <br> Automatically fetches and processes video subtitles for accurate contextual responses. <br>

🧠 AI-Powered Conversational Interface: <br> Powered by Gemini LLM and LangChain, delivering human-like and detailed responses. <br>

🔍 Source Document Retrieval: <br> Links to the referenced YouTube videos are provided alongside chatbot answers. <br>

💾 Chat History: <br> Save and revisit your conversations seamlessly for future reference. <br>

🗣️ Text-to-Speech: <br> Converts responses to audio for accessibility. <br>

🛠️ Feedback Integration: <br> Users can rate the responses, allowing continuous improvement based on feedback. <br>

<br>
Core Technologies
LangChain: For embedding creation, document retrieval, and conversational chains. <br>
Gemini API: Provides state-of-the-art language understanding and generation capabilities. <br>
Streamlit: For building the user-friendly web interface. <br>
FAISS: For efficient document similarity search. <br>
Sentence-Transformers: For text embeddings. <br>
YouTube Transcript API: For fetching video subtitles. <br>
<br>
How It Works
Subtitle Extraction: <br> Subtitles are fetched from YouTube videos and processed into manageable chunks using text-splitting techniques. <br>

Embedding Creation: <br> Text embeddings are generated using sentence-transformers, and a vector database is built using FAISS for quick retrieval. <br>

Conversational Chain: <br> Queries are processed through LangChain's conversational chain, and the Gemini LLM generates natural, context-aware answers. <br>

Chat Interface: <br> Users interact with the chatbot via a Streamlit-based web app, supporting rich features like chat history, text-to-speech, and feedback. <br>

<br>
Setup and Installation
Clone the repository. <br>
Install dependencies using: <br>
bash
Copy code
pip install -r requirements.txt  
Set up the Gemini API key in your environment: <br>
bash
Copy code
export GEMINI_API_KEY="your_api_key"  
Run the Streamlit application: <br>
bash
Copy code
streamlit run app.py  
<br>
Reference
This bot has been trained using videos and subtitles from Nikhil Kamath's YouTube channel, providing a rich dataset for creating an engaging conversational experience.

<br>
Future Improvements
Support for multiple languages in subtitles and queries. <br>
Integration with more video platforms. <br>
Enhanced GUI features for better user experience. <br>
