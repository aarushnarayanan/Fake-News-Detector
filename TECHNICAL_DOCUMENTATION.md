# 🦅 VeriScope: Technical Reference Manual

VeriScope is a high-performance, containerized AI-as-a-Service (AIaaS) application designed to detect misinformation and political bias in news articles using deep learning. This document serves as the definitive technical guide for developers, maintainers, and AI models.

---

## 🏗️ 1. System Architecture
VeriScope follows a decoupled, multi-tier architecture designed for scalability and resiliency.

*   **Client Tier**: React Single-Page Application (SPA) hosted on **AWS S3** as a static website.
*   **API Tier**: Asynchronous FastAPI server running in a **Docker** container on **AWS ECS Fargate**.
*   **Intelligence Tier**: **RoBERTa-base** Transformer model loaded in-memory within the backend container.
*   **Data Tier**: Persistent **PostgreSQL** instance managed by **AWS RDS**.
*   **Extraction Tier**: **Trafilatura** engine utilized for precise, real-time web content scraping.

---

## 🌍 2. Frontend Infrastructure (React)
The frontend is built for visual impact and high responsiveness.

*   **Core Tech**: React 19, Vite, Tailwind CSS, Framer Motion (for animations).
*   **Key Routing**: Uses `react-router-dom` with three primary views:
    *   `HomePage.jsx`: Dual-mode input for raw text and article URLs.
    *   `ResultsPage.jsx`: Dynamic "Fake Meter" visualization and snippet highlighting.
    *   `HistoryPage.jsx`: Personal analysis log with interactive feedback.
*   **State Management**: 
    *   **Local History**: Leverages browser `localStorage` to provide instant, zero-latency access to past analysis without hitting the server.
    *   **Config**: `src/config.js` acts as the single source of truth for the production API endpoint.

---

## 🧠 3. Backend & AI Logic (FastAPI + RoBERTa)
The backend is a high-concurrency Python engine.

*   **Model**: **RoBERTa-base** fine-tuned for fake news detection.
    *   **Singleton Loading**: The model and tokenizer are initialized once on startup and globally cached using a singleton class pattern to ensure zero memory duplication.
    *   **Lifespan Management**: Uses FastAPI lifespan events to manage hardware resources (MPS/CPU) and prepopulate the model cache.
    *   **Threshold**: Optimized at **0.60** for improved sensitivity to satire and nuanced misinformation.
    *   **Inference Loop**: Articles are broken into overlapping snippets; each snippet is scored, and a weighted probability is calculated for the entire document.
*   **Endpoints**:
    *   `POST /analyze`: The primary inference engine. Accepts `title` and `text`.
    *   `POST /scrape`: Ingests a URL, extracts the "clean" article body, and returns metadata.
    *   `GET /history`: Retrieves the global log of all verified predictions.
    *   `POST /feedback`: Records user corrections to the database for future fine-tuning.
*   **Middlewares**:
    *   **CORS**: Configured to restrict requests to the authorized S3 domain.
    *   **SlowAPI**: Implements IP-based rate limiting (5 requests/minute) to prevent DoS.

---

## 🗄️ 4. Data Architecture (RDS PostgreSQL)
VeriScope uses a relational schema to track long-term trends and user feedback.

### **Table: `predictions`**
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Primary Key. |
| `input_text_hash` | String | SHA-1 hash of the text to prevent redundant processing. |
| `title` | String | Optional article headline. |
| `input_text_preview` | String | First 100 characters for history browsing. |
| `label` | String | AI Label ("REAL" or "FAKE"). |
| `probability` | Float | Confidence score (0.0 to 1.0). |
| `created_at` | DateTime | Timestamp of the analysis. |

### **Table: `feedback`**
| Field | Type | Description |
| :--- | :--- | :--- |
| `prediction_id` | Integer | Foreign Key to the prediction table. |
| `user_label` | String | The user's corrected label. |
| `comments` | Text | Optional user notes. |

---

## 🐳 5. Containerization & DevOps
The project uses a "Write Once, Run Anywhere" philosophy via Docker.

*   **Docker Compose**: Orchestrates `frontend`, `backend`, and `db` services for local development. Includes **Volumes** for hot-reloading code.
*   **Production Build**: 
    *   Backend is built for **Linux/AMD64** (required for AWS Fargate) despite local ARM64 development.
    *   Frontend is built into a static `dist/` folder and synchronized to S3.
*   **CI/CD Baseline**: AWS CLI is used to synchronize the `dist/` folder and trigger ECS "Force New Deployments" upon image updates.

---

## 🛡️ 6. Security & Stability Protocols
*   **Input Sanitization**: All incoming text is stripped of HTML tags using regex to prevent script injection.
*   **Defensive Rendering**: The frontend uses `typeof` checks and `JSON.stringify` safeguards to ensure that unexpected API errors never cause a "Black Screen" crash.
*   **Character Limits**: Input is capped at 3,000 characters to ensure model stability and prevent resource exhaustion.

---

## 🚀 7. Launch Guide (Developer Onboarding)

### **Local Launch**
1.  Clone the repository.
2.  Ensure Docker Desktop is running.
3.  Execute: `docker-compose up --build`.

**Production Push (AWS)**
1.  **Sync Code**: Push latest logic to your Git repository.
2.  **Server Update**: On the AWS instance, pull latest code and execute `docker compose up --build -d`.
3.  **Dynamic Config**: The `frontend/src/config.js` will automatically detect the AWS environment and route to the production backend.
4.  **Static Deploy**: If frontend changes were made, run `npm run build` and sync to S3: `aws s3 sync dist/ s3://[YOUR_BUCKET] --delete`.

---

## 📈 8. Future Roadmap
*   **Active Learning**: Implement an automated pipeline to pull "Incorrect" predictions from the `feedback` table and fine-tune the RoBERTa model.
*   **Stylometric Analysis**: Incorporate metadata such as punctuation density and emotional sentiment (clickbait detection) as additional input features.

---

**End of Technical Documentation**
