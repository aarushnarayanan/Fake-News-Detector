# VeriScope - Full Stack AI Application

**VeriScope** is a production-grade web application that uses deep learning (RoBERTa) to detect failing news and misinformation. It features a modern React frontend hosted on AWS S3, a FastAPI backend on AWS ECS Fargate, and a PostgreSQL database on AWS RDS.

## 🚀 How It Works (The Flow)

1.  **User Interaction**: A user visits the website and pastes a news article.
2.  **API Request**: The React Frontend sends the text to the Backend API (`POST /analyze`).
3.  **Preprocessing**: The Backend cleans the text and prepares it for the AI model.
4.  **AI Analysis**: The `FakeNewsPredictor` (RoBERTa model) analyzes the text, assigns a probability score, and highlights suspicious sentences ("evidence").
5.  **Data Storage**: The result is saved to the PostgreSQL database for history tracking.
6.  **Response**: The Backend sends the results (Label, Confidence, Highlights) back to the Frontend.
7.  **Visualization**: The Frontend displays a gauge, highlighted text, and top suspicious snippets.

---

## 📂 Project Structure & File Guide

### 1. Backend (`/backend`)
Handles the logic, AI model, and database connections.

*   **`main.py`**: The entry point of the API. It defines the URL endpoints:
    *   `/analyze`: Runs the AI model.
    *   `/history`: Fetches past predictions.
    *   `/feedback`: Saves user corrections.
*   **`database.py`**: Manages the connection to PostgreSQL. Defines the `Prediction` and `Feedback` table structures (SQLAlchemy Models).
*   **`requirements.txt`**: Lists all python libraries needed (e.g., `fastapi`, `torch`, `transformers`).
*   **`Dockerfile`**: Instructions for Docker to package this backend into a container image for AWS.

### 2. AI Model (`/models`)
The brain of the operation.

*   **`predict.py`**: The core AI logic.
    *   Class `FakeNewsPredictor`: Loads the RoBERTa model.
    *   Method `predict(text)`: Returns `FAKE` or `REAL`, confidence score, and highlighted evidence.
*   **`model_setup/`**: (Generated) Contains the saved RoBERTa model files (`pytorch_model.bin`, `config.json`). *Note: These large files are usually git-ignored.*

### 3. Frontend (`/frontend`)
The user interface built with React, Vite, and TailwindCSS.

*   **`src/config.js`**: **Crucial Configuration**. This file holds the `API_BASE_URL` pointing to the live AWS Backend.
*   **`src/pages/`**:
    *   **`HomePage.jsx`**: The landing page with the input text area.
    *   **`ResultsPage.jsx`**: Displays the AI results, confidence gauge, and highlighted text.
    *   **`HistoryPage.jsx`**: Table view of previous analyses.
*   **`src/components/Navbar.jsx`**: The navigation bar.
*   **`vite.config.js`**: Configuration for the build tool (Vite).
*   **`Dockerfile`**: Instructions to package the frontend (using Nginx) - *Used for local docker-compose testing*.

### 4. Infrastructure (Root)
*   **`docker-compose.yml`**: Allows running the entire stack (Backend + Frontend + Database) locally with one command (`docker-compose up`).
*   **`manual_aws_walkthrough.md`**: Step-by-step guide on how this specific deployment was built on AWS.

---

## 🛠️ Operational Guide

### Running Locally (Development)
1.  **Backend** (Terminal 1):
    ```bash
    pip install -r backend/requirements.txt
    uvicorn backend.main:app --reload
    ```
2.  **Frontend** (Terminal 2):
    ```bash
    cd frontend
    npm run dev
    ```

### Running Locally (Docker)
```bash
docker-compose up --build
```
*   Frontend: `http://localhost:5173`
*   Backend: `http://localhost:8000`

### Deployment (AWS Production)
*   **Frontend**: Hosted as a **Static Website on S3**. Files are uploaded from `frontend/dist`.
*   **Backend**: Hosted as a **Container on ECS Fargate**. Docker image is stored in ECR.
*   **Database**: Hosted on **RDS PostgreSQL**.
*   **Networking**: Security Groups allow traffic between the Backend Service and the Database.

---

## 🔐 Security Features
*   **Rate Limiting**: Limits users to 5 requests/minute to prevent abuse (`slowapi`).
*   **Input Sanitization**: Cleans HTML tags from input to prevent XSS attacks.
*   **CORS**: Only allows browser requests from authorized domains.
