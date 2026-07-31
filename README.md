# AI-chat
🤖 AI Document – an Antigravity-based agent that generates and keeps your project docs (API reference, architecture overview, README) up to date automatically.
AI Document Assistant 🤖📄
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/4521849e-61d1-469b-906d-a7582967f5aa" />

An AI-powered chatbot that allows users to upload documents and ask questions in natural language. The application uses Google Gemini, LangChain, and Streamlit to analyze uploaded files and generate accurate, context-aware answers using Retrieval-Augmented Generation (RAG).

🚀 Features
📂 Upload PDF, DOCX, and TXT documents
🤖 AI-powered chatbot using Google Gemini
🔍 Retrieval-Augmented Generation (RAG) for context-aware responses
💬 Interactive chat interface built with Streamlit
⚡ Fast document processing and semantic search
🔐 Secure API key configuration
🛠️ Tech Stack
Python
Streamlit
LangChain
Google Gemini API
FAISS (Vector Database)
Google Generative AI Embeddings
PyPDF2
python-docx
📂 Project Structure
AI-Document-Assistant/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Project dependencies
│── .env                   # API Key (not uploaded to GitHub)
│── README.md
│── utils.py               # Helper functions (if applicable)
│── documents/             # Uploaded files (optional)
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/AI-Document-Assistant.git
cd AI-Document-Assistant
2. Create a Virtual Environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Add Your Gemini API Key

Create a .env file and add:

GOOGLE_API_KEY=your_gemini_api_key
5. Run the Application
streamlit run app.py
💡 How It Works
Enter your Google Gemini API key.
Upload a PDF, DOCX, or TXT file.
Click Process Documents.
Ask questions about the uploaded document.
Receive AI-generated answers based on the document's content.
📸 Screenshot

Add a screenshot of the application here.

![AI Document Assistant](images/screenshot.png)
🔮 Future Improvements
Support multiple document uploads
Chat history
Conversation memory
Multiple LLM support
OCR for scanned PDFs
Voice input
Document summarization
🤝 Contributing

Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Annapoorneshwari

GitHub: https://github.com/your-annapoorneshwari
