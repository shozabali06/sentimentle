import { Router, type Request, type Response } from 'express';
import { analyzeSentiment } from '../services/sentiment.service.js';
import { sentimentLimiter } from '../middlewares/rateLimiter.js';

const router = Router();

// Health check route
router.get('/health', (req: Request, res: Response) => {
    res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Analyze text route
// Uses sentimentLimiter middleware to limit the number of requests
router.post('/analyze-text', sentimentLimiter, async (req: Request, res: Response) => {
    const { text } = req.body as { text: string };

    // Check if the text is required and cannot be empty
    if (!text || text.trim() === '') {
        return res.status(400).json({ error: 'Text is required and cannot be empty.' });
    }

    try {
        // Analyze the sentiment of the text
        const sentiment = await analyzeSentiment(text);
        // Return the sentiment of the text
        res.json({
            text: text,
            sentiment: sentiment
        });
    } catch (error) {
        // If the error is an instance of Error, log the error message and return a 500 error
        if (error instanceof Error) {
            console.error('Error during /analyze-text process:', error.message);
            res.status(500).json({ error: 'Failed to analyze the text.' });
        } else {
            // If the error is not an instance of Error, log the error and return a 500 error
            console.error('An unknown error occurred:', error);
            res.status(500).json({ error: 'An unknown error occurred.' });
        }
    }
});

export default router;