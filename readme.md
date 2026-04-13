# 🚀 TubeGPT – AI-Powered YouTube RAG Assistant

TubeGPT is a scalable **RAG-based AI assistant** that allows users to interact with YouTube video content using natural language.
It extracts transcripts, performs semantic search using a vector database, and generates intelligent, context-aware responses using **Google Gemini**.

---

## 🔥 Features

* 🎥 YouTube transcript extraction & processing
* 🧠 Semantic search using Pinecone vector database
* 🤖 AI-generated responses using Gemini API
* 💬 ChatGPT-like conversational experience
* 🧾 Chat history memory (thread-based system)
* 🎯 Intelligent YouTube video recommendations
* 🔐 Secure authentication using Keycloak (JWT + RBAC)
* ⚡ High-performance FastAPI backend
* 🛡️ Prompt-injection & jailbreak protection

---

## ⚙️ Tech Stack

* **Backend:** Python, FastAPI
* **AI Model:** Google Gemini (via Service Account)
* **Vector DB:** Pinecone
* **Embedding Model:** Sentence Transformers
* **Databases:**

  * PostgreSQL → Chat threads & history
  * MongoDB → User profile & metadata
* **Authentication:** Keycloak (OAuth2 / OpenID Connect)
* **Architecture:** RAG (Retrieval-Augmented Generation)

---

## 🧠 System Architecture

```text
[ User ]
   │
   ▼
[ Keycloak Authentication ]
   │ (JWT Token)
   ▼
[ FastAPI Backend ]
   │
   ├── Chat History (PostgreSQL)
   ├── User Profile (MongoDB)
   │
   ▼
[ RAG Pipeline ]
   │
   ├── YouTube Transcript Extraction
   ├── Chunking
   ├── Embedding (SentenceTransformer)
   ▼
[ Pinecone Vector DB ]
   │
   ▼
[ Context Retrieval ]
   │
   ▼
[ Gemini API (LLM) ]
   │
   ▼
[ AI Response + Recommendations ]
   │
   ▼
[ User ]
```

---

## 🔄 How It Works

1. User authenticates via **Keycloak** (JWT-based)
2. YouTube transcripts are extracted and processed
3. Transcripts are chunked and converted into embeddings
4. Embeddings are stored in **Pinecone**
5. User sends a query
6. System retrieves relevant context from vector DB
7. Context + chat history is sent to **Gemini**
8. AI generates response
9. If similarity threshold > 0.4 → video recommendations shown

---

## 💬 Chat Flow

* Thread-based conversation system
* Context built using previous interactions
* Follow-up questions generated automatically
* Chat behaves like ChatGPT with memory

---

## 🔐 Authentication (Keycloak)

* JWT-based authentication
* Middleware handles:

  * Token validation
  * Token refresh
  * User identity extraction
* Role-based access control (RBAC) supported

---

## 📌 API Endpoints

* `POST /chat` → Main chatbot endpoint
* `POST /ingest` → Ingest YouTube videos
* `GET /history/{thread_id}` → Fetch chat history
* `GET /threads/{user_id}` → User conversations
* `POST /ai/start` → Start AI conversation
* `POST /ai/continue` → Continue conversation
* `GET /health` → Health check

---

#