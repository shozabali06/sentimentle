import express from 'express';
import cors from 'cors';
import analysisRoutes from './routes/analysis.routes.js';
import { apiLimiter } from './middlewares/rateLimiter.js';

const app = express();

const corsOptions = {
    origin: process.env.FRONTEND_URL || 'http://localhost:5173',
    credentials: true,
    optionsSuccessStatus: 200,
}
app.use(cors(corsOptions));

app.use(express.json());

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'sentimentle server is running',
        environment: process.env.NODE_ENV || 'development',
        timestamp: new Date().toISOString()
    });
});

app.use('/api/', apiLimiter);

app.use('/api/', analysisRoutes);

export default app;