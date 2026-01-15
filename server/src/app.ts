import express from 'express';
import cors from 'cors';
import analysisRoutes from './routes/analysis.routes.js';
import { apiLimiter } from './middlewares/rateLimiter.js';

// Create a new express application
const app = express();

// Configure CORS options
const corsOptions = {
    // Set the origin of the request to the frontend URL
    origin: process.env.FRONTEND_URL || 'http://localhost:5173',
    // Allow credentials to be sent in the request
    credentials: true,
    // Set the status code for the response to 200
    optionsSuccessStatus: 200,
}
// Use the CORS middleware to allow requests from the frontend URL
app.use(cors(corsOptions));

// Use the express.json middleware to parse the request body as JSON
app.use(express.json());

// Root endpoint to check if the server is running
app.get('/', (req, res) => {
    // Return a JSON response with the message, environment, and timestamp
    res.json({
        message: 'sentimentle server is running',
        environment: process.env.NODE_ENV || 'development',
        timestamp: new Date().toISOString()
    });
});

// Use the apiLimiter middleware to limit the number of requests
app.use('/api/', apiLimiter);

// Use the analysisRoutes to handle the analysis requests
app.use('/api/', analysisRoutes);

export default app;