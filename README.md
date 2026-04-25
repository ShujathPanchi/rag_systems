# rag_systems
Project to build RAG system
# Cognivault AI — One Page Product & Engineering Document

## Executive Summary

Cognivault AI is a private enterprise knowledge assistant designed to help organizations unlock value from internal documents through AI powered search, retrieval, summarization, and decision support. Built using Streamlit, a local LLM runtime, and vector search, the platform transforms static files into an intelligent conversational workspace where employees can ask questions and receive trusted answers grounded in company knowledge.

## Business Problem Solved

Most organizations store policies, architecture documents, onboarding guides, SOPs, and operational knowledge across disconnected files and folders. Employees lose time searching for information, duplicate work increases, and decisions are delayed. Cognivault AI centralizes that knowledge into one secure interface and enables natural language access to information instantly.

## Product Vision

Create an internal AI workspace that feels as intuitive as ChatGPT but is powered by enterprise data, controlled privately, and tailored for business workflows.

## Core Capabilities

### 1. AI Chat Workspace

Users ask natural language questions such as leave policy queries, onboarding steps, technical architecture questions, or leadership summaries. The assistant retrieves relevant internal knowledge and generates contextual responses.

### 2. Retrieval Augmented Generation (RAG)

Uploaded files are chunked, embedded, indexed, and searched at runtime. Relevant chunks are passed to the model so answers are based on company content rather than generic model memory.

### 3. Assistant Modes

Different user personas receive tailored responses:

* Onboarding: beginner friendly guidance
* Support: troubleshooting focused answers
* Architect: systems and dependency explanations
* Leadership: business impact and executive summaries

### 4. Governance Center

Operational controls for knowledge management:

* Upload files
* Delete files
* Rebuild search index
* Track file count
* Track indexed chunks
* Maintain content freshness

### 5. Persistent Chat Memory

Conversation history is stored locally so users can continue previous sessions, switch tabs, and clear chat when required.

### 6. Premium Enterprise UX

Modern product design with branded sidebar, hero banner, quick actions, KPI cards, system health indicators, dynamic placeholders, and workspace style layout.

## Technical Architecture

Frontend uses Streamlit. Business logic is split into modular Python services. Vector retrieval uses FAISS or equivalent store. Local LLM inference runs through Ollama. Session data is stored in lightweight local JSON. Styling uses custom CSS.

## Final Modular Structure

* app.py for routing and startup
* components for reusable UI blocks
* views for Chat and Governance screens
* services for model, RAG, vector store, session memory
* styles for custom UI
* data folder for uploaded documents
* system folder for chat history and metadata

## Engineering Journey

The product evolved from a single file prototype into a modular enterprise application through multiple phases: prototype chat assistant, document retrieval engine, governance dashboard, premium UI redesign, persistent memory, workspace UX optimization, and product grade controls.

## Value to Organizations

* Faster knowledge discovery
* Reduced support load
* Better onboarding speed
* Stronger decision making
* Secure private AI adoption
* Reuse of existing documents

## Future Roadmap

Authentication, role based access, multi user memory, analytics, streaming responses, API integrations, cloud deployment, Teams or Slack integration, and admin reporting.

## GitHub Release Summary

Built Cognivault AI, a private enterprise knowledge assistant with RAG search, governance dashboard, persistent memory, modular architecture, premium UX, dynamic assistant modes, and local LLM integration.

## Author Note

Designed and iterated by Sujath with a product first mindset focused on enterprise AI usability, governance, and real world adoption.

=====================================================================================================================================================

# Cognivault AI — End to End Project Setup Guide

## Overview

This document provides the complete setup journey for Cognivault AI from initial environment preparation to final launch. It includes project structure, commands, dependencies, configuration, indexing flow, and run instructions.

## 1. Prerequisites

Install the following before starting:

* Python 3.10 or above
* Git
* VS Code or preferred IDE
* Ollama installed locally
* Internet connection for package installation

## 2. Create Project Folder

```bash
mkdir cognivault-ai
cd cognivault-ai
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment (Windows)

```bash
venv\Scripts\activate
```

### Activate Environment (Mac/Linux)

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install streamlit
pip install faiss-cpu
pip install sentence-transformers
pip install pypdf
pip install langchain
pip install numpy
pip install pandas
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

## 5. Install and Run Local LLM

Install Ollama from official website.

Run model:

```bash
ollama run mistral
```

Optional models:

```bash
ollama run llama3
ollama run phi3
```

## 6. Create Final Folder Structure

```text
cognivault-ai/
│── app.py
│── styles_loader.py
│── requirements.txt
│
├── components/
│   ├── hero.py
│   └── sidebar.py
│
├── views/
│   ├── chat.py
│   └── governance.py
│
├── services/
│   ├── model_service.py
│   ├── rag_service.py
│   ├── vector_store.py
│   └── session_store.py
│
├── styles/
│   └── main.css
│
├── data/
├── system/
└── vector_db/
```

## 7. Core Files Responsibility

### app.py

Main router and app bootstrap.

### sidebar.py

Navigation + assistant mode controls.

### chat.py

Main AI workspace.

### governance.py

Upload files, delete files, rebuild index.

### vector_store.py

Chunking, embeddings, FAISS load/save.

### rag_service.py

Retrieve top chunks + generate answer.

### session_store.py

Save and load chat history.

## 8. Create Data Folders

```bash
mkdir data
mkdir system
mkdir vector_db
```

## 9. Add Styling Loader

Load custom CSS from styles/main.css into Streamlit.

## 10. Build Governance Flow

Upload files into data folder.

Supported formats:

* PDF
* TXT

Actions:

* Upload files
* Delete files
* Rebuild index
* Show chunk counts

## 11. Indexing Pipeline

When files uploaded:

1. Read file text
2. Split into chunks
3. Generate embeddings
4. Save vectors in FAISS
5. Save metadata texts

## 12. Chat Flow

When user asks question:

1. Load vector index
2. Embed question
3. Retrieve top relevant chunks
4. Send chunks + prompt to local model
5. Display answer
6. Save chat history

## 13. Run Application

```bash
streamlit run app.py
```

Default browser opens locally.

## 14. Common Commands

### Freeze requirements

```bash
pip freeze > requirements.txt
```

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Stop Streamlit

```bash
Ctrl + C
```

## 15. Session Persistence

Chat stored in:

```text
system/chat_history.json
```

Allows:

* reload previous chats
* retain sessions
* clear memory manually

## 16. Git Setup

```bash
git init
git add .
git commit -m "Initial Cognivault AI release"
```

Add remote:

```bash
git remote add origin <repo-url>
git push -u origin main
```

## 17. Recommended .gitignore

```text
venv/
__pycache__/
*.pyc
system/chat_history.json
vector_db/
.streamlit/
```

## 18. Troubleshooting

### Port already used

```bash
streamlit run app.py --server.port 8502
```

### Module not found

```bash
pip install -r requirements.txt
```

### Ollama not responding

Ensure model running:

```bash
ollama run mistral
```

### Empty answers

Rebuild index after uploading files.

## 19. Final Product Features Delivered

* AI chat workspace
* Governance center
* Persistent memory
* Dynamic placeholders
* Assistant modes
* Premium UI
* Local secure AI
* Vector search

## 20. Future Enhancements

* User login
* Multi-user memory
* Cloud hosting
* API integrations
* Slack / Teams bot
* Analytics dashboard

## Final Launch Command

```bash
streamlit run app.py
```

## Author

Built by Sujath as an enterprise AI product from concept to execution.
