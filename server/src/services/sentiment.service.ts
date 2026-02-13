import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Get the filename and directory name of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve paths relative to project root (go up from server/dist/services to project root)
const projectRoot = path.resolve(__dirname, '..', '..', '..');

// Get the path to the Python executable
const pythonExecutablePath = process.env.NODE_ENV === 'production'
    // For production, use the Python executable in the ml-env directory
    ? path.join(projectRoot, 'ml', 'ml-env', 'bin', 'python') 
    // Check if the platform is Windows
    : process.platform === 'win32'
    // For Windows, use the Python executable in the Scripts directory
        ? path.join(projectRoot, 'ml', 'ml-env', 'Scripts', 'python.exe')
        // For other platforms, use the Python executable in the bin directory
        : 'python';

// Get the path to the Python script (predict.py)
const pythonScriptPath = path.join(projectRoot, 'ml', 'scripts', 'predict.py');

export interface SentimentAnalysisResult {
    sentiment: string;
    confidence: number;
}

// Analyze the sentiment of the text; returns { sentiment, confidence } from Python script JSON output
export function analyzeSentiment(text: string): Promise<SentimentAnalysisResult> {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn(pythonExecutablePath, [pythonScriptPath, text], {
            cwd: projectRoot
        });

        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => { result += data.toString(); });
        pythonProcess.stderr.on('data', (data) => { error += data.toString(); });

        pythonProcess.on('close', (code) => {
            if (error) return reject(new Error(`Python script error: ${error}`));
            if (code !== 0) return reject(new Error(`Python script exited with code ${code}`));

            try {
                const parsed = JSON.parse(result.trim()) as SentimentAnalysisResult;
                if (typeof parsed.sentiment !== 'string' || typeof parsed.confidence !== 'number') {
                    return reject(new Error('Invalid sentiment script output: missing sentiment or confidence'));
                }
                resolve(parsed);
            } catch {
                reject(new Error(`Invalid sentiment script output (not JSON): ${result.trim()}`));
            }
        });

        pythonProcess.on('error', (err) => {
            reject(new Error(`Failed to start prediction service: ${err.message}. Python: ${pythonExecutablePath}, Script: ${pythonScriptPath}`));
        });
    });
}