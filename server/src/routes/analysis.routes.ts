import { Router, type Request, type Response } from 'express';
import { analyzeSentiment } from '../services/sentiment.service.js';
import { sentimentLimiter } from '../middlewares/rateLimiter.js';

const router = Router();

router.get('/health', (req: Request, res: Response) => {
    res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

router.post('/analyze-text', sentimentLimiter, async (req: Request, res: Response) => {
    const { text } = req.body as { text: string };

    if (!text || text.trim() === '') {
        return res.status(400).json({ error: 'Text is required and cannot be empty.' });
    }

    try {
        const sentiment = await analyzeSentiment(text);
        res.json({
            text: text,
            sentiment: sentiment
        });
    } catch (error) {
        if (error instanceof Error) {
            console.error('Error during /analyze-text process:', error.message);
            res.status(500).json({ error: 'Failed to analyze the text.' });
        } else {
            console.error('An unknown error occurred:', error);
            res.status(500).json({ error: 'An unknown error occurred.' });
        }
    }
});

export default router;