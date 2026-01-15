# Sentimentle - Real-Time Text Sentiment Analyzer

A full-stack web application for real-time text sentiment analysis using machine learning. Built with React, Node.js, Express, and Python (scikit-learn).

## 🎯 Features

- **Real-time Sentiment Analysis**: Analyze text sentiment instantly using a trained ML model
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- **Dark/Light Mode**: Theme support with system preference detection
- **Rate Limiting**: API protection with request rate limiting
- **RESTful API**: Clean, well-structured backend API
- **Docker Support**: Easy deployment with Docker containerization

## 🏗️ Architecture

This project follows a microservices-like architecture with three main components:

- **Frontend**: React + TypeScript + Vite application
- **Backend**: Node.js + Express + TypeScript API server
- **ML Service**: Python script using scikit-learn for sentiment prediction

### Tech Stack

**Frontend:**
- React 19
- TypeScript
- Vite
- Tailwind CSS 4
- Radix UI components
- Lucide React icons

**Backend:**
- Node.js
- Express 5
- TypeScript
- Express Rate Limit
- CORS

**Machine Learning:**
- Python 3
- scikit-learn
- NLTK
- pandas
- numpy
- joblib

## 📁 Project Structure

```
setimentle/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── lib/             # Utility functions
│   │   └── App.tsx          # Main app component
│   ├── package.json
│   └── vite.config.ts
├── server/                   # Node.js backend server
│   ├── src/
│   │   ├── routes/          # API routes
│   │   ├── services/        # Business logic
│   │   ├── middlewares/     # Express middlewares
│   │   ├── app.ts           # Express app setup
│   │   └── server.ts        # Server entry point
│   ├── dist/                # Compiled JavaScript
│   └── package.json
├── ml/                       # Machine learning service
│   ├── scripts/
│   │   └── predict.py       # Sentiment prediction script
│   ├── models/              # Trained ML models
│   ├── datasets/            # Training datasets
│   ├── notebooks/           # Jupyter notebooks
│   └── requirements.txt
├── DockerFile               # Docker configuration
└── package.json             # Root package.json with scripts
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v20 or higher)
- **Python** (3.8 or higher)
- **npm** or **yarn**
- **pip**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shozabali06/sentimentle
   cd "sentimentle"
   ```

2. **Install all dependencies**
   ```bash
   npm run install:all
   ```
   
   This will install dependencies for:
   - Root project
   - Frontend
   - Backend server

3. **Set up Python virtual environment**
   ```bash
   cd ml
   python -m venv ml-env
   ```

4. **Activate virtual environment**
   
   **Windows:**
   ```bash
   ml-env\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source ml-env/bin/activate
   ```

5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Download NLTK data** (required for text preprocessing)
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

### Running the Application

#### Development Mode

1. **Start the backend server** (Terminal 1)
   ```bash
   npm run dev:server
   ```
   Server runs on `http://localhost:3000`

2. **Start the frontend** (Terminal 2)
   ```bash
   npm run dev:frontend
   ```
   Frontend runs on `http://localhost:5173`

#### Production Mode

1. **Build the backend**
   ```bash
   cd server
   npm run build
   npm start
   ```

2. **Build the frontend**
   ```bash
   cd frontend
   npm run build
   npm run preview
   ```

## 🐳 Docker Deployment

Build and run the application using Docker:

```bash
docker build -t sentimentle .
docker run -p 3000:3000 sentimentle
```

The Dockerfile:
- Uses Node.js 20-slim base image
- Installs Python 3 and dependencies
- Sets up the ML environment
- Builds and runs the Node.js server

## 📡 API Endpoints

### Base URL
- Development: `http://localhost:3000`
- Production: Set via `PORT` environment variable

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "sentimentle server is running",
  "environment": "development",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### `GET /api/health`
API health check.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### `POST /api/analyze-text`
Analyze text sentiment.

**Request Body:**
```json
{
  "text": "I love this product! It's amazing."
}
```

**Response:**
```json
{
  "text": "I love this product! It's amazing.",
  "sentiment": "positive"
}
```

**Possible Sentiment Values:**
- `positive`
- `negative`
- `neutral`

**Rate Limiting:** This endpoint is rate-limited to prevent abuse.

## 🔧 Configuration

### Environment Variables

**Backend (`server/`):**
- `PORT`: Server port (default: `3000`)
- `NODE_ENV`: Environment (`development` or `production`)
- `FRONTEND_URL`: Frontend URL for CORS (default: `http://localhost:5173`)

**Example `.env` file:**
```env
PORT=3000
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

### ML Model

The application uses pre-trained models located in `ml/models/`:
- `sentiment_model_v3.pkl`: Trained sentiment classification model
- `vectorizer_v3.pkl`: TF-IDF vectorizer for text preprocessing

The ML script (`ml/scripts/predict.py`) handles:
- Text preprocessing (lowercasing, noise removal, stemming)
- Stopword removal (preserving negation words)
- Sentiment prediction using the loaded model

## 🧪 Development

### Project Scripts

**Root:**
- `npm run dev:frontend`: Start frontend dev server
- `npm run dev:server`: Start backend dev server
- `npm run install:all`: Install all dependencies

**Frontend:**
- `npm run dev`: Start Vite dev server
- `npm run build`: Build for production
- `npm run preview`: Preview production build
- `npm run lint`: Run ESLint

**Backend:**
- `npm run dev`: Start with nodemon (auto-reload)
- `npm run build`: Compile TypeScript
- `npm start`: Start production server

## 📝 Notes

- The ML models must be present in `ml/models/` directory
- The Python virtual environment path is configured differently for Windows vs. Unix systems
- Rate limiting is applied to the `/api/` endpoints to prevent abuse
- CORS is configured to allow requests from the frontend URL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

ISC

## 🙏 Acknowledgments

- Built with React, Node.js, and scikit-learn
- UI components from Radix UI
- Icons from Lucide React

---

Made with ❤️ for sentiment analysis

