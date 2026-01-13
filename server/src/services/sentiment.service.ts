import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve paths relative to project root (go up from server/dist/services to project root)
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const pythonExecutablePath = process.env.NODE_ENV === 'production'
    ? path.join(projectRoot, 'ml', 'ml-env', 'bin', 'python')
    : process.platform === 'win32'
        ? path.join(projectRoot, 'ml', 'ml-env', 'Scripts', 'python.exe')
        : 'python';

const pythonScriptPath = process.env.NODE_ENV === 'production' ? path.join(projectRoot, 'app', 'ml', 'scripts', 'predict.py') : path.join(projectRoot, 'ml', 'scripts', 'predict.py');

export function analyzeSentiment(text: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn(pythonExecutablePath, [pythonScriptPath, text], {
            cwd: projectRoot // Set working directory to project root for relative paths in Python script
        });

        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => { result += data.toString(); });
        pythonProcess.stderr.on('data', (data) => { error += data.toString(); });

        pythonProcess.on('close', (code) => {
            if (error) return reject(new Error(`Python script error: ${error}`));

            if (code !== 0) return reject(new Error(`Python script exited with code ${code}`));

            resolve(result.trim());
        });

        pythonProcess.on('error', (err) => {
            reject(new Error(`Failed to start prediction service: ${err.message}. Python: ${pythonExecutablePath}, Script: ${pythonScriptPath}`));
        });
    });
}