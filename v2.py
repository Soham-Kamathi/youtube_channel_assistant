# Install required packages if not already installed
#
#pip install google-generativeai
#pip install langchain faiss-cpu sentence-transformers huggingface_hub
#pip install langchain langchain-community faiss-cpu sentence-transformers huggingface_hub

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import os
from tqdm.notebook import tqdm
import torch
import gc
from tqdm import tqdm
from langchain.prompts import PromptTemplate
# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyD4qGBhnLPy5kFqdK62vfUhkkzgJXcahXI"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping
import google.generativeai as genai

from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping
import google.generativeai as genai
from pydantic import Field

import json
from datetime import datetime

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    model: Any = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

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

class SubtitleChatbot:
    def __init__(self, subtitles_dir='youtube_subtitles'):
        self.subtitles_dir = subtitles_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Add version check
        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
            import keras
            print(f"Keras version: {keras.__version__}")
        except ImportError:
            print("TensorFlow/Keras not properly installed")

    def load_documents(self):
        """Load all subtitle files from the directory with progress bar."""
        print("Loading documents...")
        files = [f for f in os.listdir(self.subtitles_dir) if f.endswith('.txt')]
        documents = []

        for file in tqdm(files, desc="Loading files"):
            try:
                loader = TextLoader(os.path.join(self.subtitles_dir, file))
                docs = loader.load()
                # Add metadata to each document
                for doc in docs:
                    # Extract video ID from filename (assuming format: <video_id>.txt)
                    video_id = file.replace('.txt', '')
                    doc.metadata['video_id'] = video_id
                    doc.metadata['video_url'] = f"https://www.youtube.com/watch?v={video_id}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        print(f"Loaded {len(documents)} documents")
        return documents

    def split_texts(self, documents):
        """Split documents into smaller chunks with progress tracking."""
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        texts = []
        for doc in tqdm(documents, desc="Splitting documents"):
            texts.extend(text_splitter.split_documents([doc]))

        print(f"Created {len(texts)} text chunks")
        return texts

    def create_embeddings(self, texts):
        """Create embeddings with batch processing and memory management."""
        print("Creating embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )

        batch_size = 64
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        vectorstore = None
        for i, batch in enumerate(tqdm(batches, desc="Creating embeddings")):
            batch_vectorstore = FAISS.from_documents(batch, embeddings)

            if vectorstore is None:
                vectorstore = batch_vectorstore
            else:
                vectorstore.merge_from(batch_vectorstore)

            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % 10 == 0:
                vectorstore.save_local("subtitle_vectorstore_checkpoint")

        print("Embeddings creation completed")
        return vectorstore

    def setup_chatbot(self, vectorstore):
        """Set up the conversational chain with Gemini."""
        print("Setting up chatbot...")

        llm = GeminiLLM()
        feedback_store = FeedbackStore()

        # Get recent negative feedback
        recent_feedback = feedback_store.feedback_data[-5:] if feedback_store.feedback_data else []
        feedback_prompt = "\n".join([
            f"Previous improvement feedback: {f['improvement_feedback']}"
            for f in recent_feedback if not f['is_helpful'] and f['improvement_feedback']
        ])

        prompt_template = f"""You are a helpful AI assistant based on youtube channel of Nikhil Kamath. 
        Previous feedback on similar responses suggests the following improvements:
        {feedback_prompt}

        Use the following pieces of context to answer the question, but do not directly quote from them. 
        Instead, synthesize the information and explain it in your own words in a clear, detailed, and comprehensive manner.

        Context: {{context}}

        Question: {{question}}

        Please provide a detailed answer that includes:
        1. A comprehensive explanation of the main points
        2. Any relevant examples or scenarios mentioned
        3. Additional context or background information
        4. Key takeaways or practical implications

        Make sure to structure your response in a clear, well-organized manner using paragraphs where appropriate. 
        Aim for a detailed response of at least 200-300 words:"""

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),  # Increased from default to get more context
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )},
            verbose=True
        )

        return qa_chain

    def load_model(self, path="subtitle_vectorstore"):
        """Load a previously trained model."""
        print(f"Loading trained model from {path}...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': self.device}
            )
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            qa_chain = self.setup_chatbot(vectorstore)
            print("Model loaded successfully!")
            return qa_chain
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def train(self):
        """Train the chatbot with progress tracking and checkpoints."""
        try:
            documents = self.load_documents()
            texts = self.split_texts(documents)
            vectorstore = self.create_embeddings(texts)

            print("Saving final vector store...")
            vectorstore.save_local("subtitle_vectorstore")

            qa_chain = self.setup_chatbot(vectorstore)
            print("Training completed successfully!")
            return qa_chain

        except Exception as e:
            print(f"Error during training: {str(e)}")
            if os.path.exists("subtitle_vectorstore_checkpoint"):
                print("Loading last checkpoint...")
                return self.load_model("subtitle_vectorstore_checkpoint")
            raise e

def chat_interface(qa_chain):
    """Simple chat interface for interacting with the bot."""
    print("\nChatbot is ready! Type 'quit' to exit.")
    while True:
        question = input("\nYou: ")
        if question.lower() == 'quit':
            break

        try:
            result = qa_chain({"question": question})
            print("\nBot:", result["answer"])

            # Show only source video links
            if "source_documents" in result:
                print("\nSource Videos:")
                # Create a set to store unique URLs
                unique_urls = set()
                
                for doc in result["source_documents"]:
                    if 'video_url' in doc.metadata:
                        unique_urls.add(doc.metadata['video_url'])
                
                # Print unique URLs
                for url in unique_urls:
                    print(f"📺 {url}")

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")

# Main execution
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("Please set your Gemini API key first!")
    else:
        print("Initializing chatbot...")
        chatbot = SubtitleChatbot()

        if os.path.exists("subtitle_vectorstore"):
            qa_chain = chatbot.load_model()
        else:
            qa_chain = chatbot.train()

        if qa_chain:
            chat_interface(qa_chain)
        else:
            print("Failed to initialize chatbot. Please check the errors above.")