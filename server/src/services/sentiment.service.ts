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

// Analyze the sentiment of the text
export function analyzeSentiment(text: string): Promise<string> {
    // Return a promise that resolves to the sentiment of the text
    return new Promise((resolve, reject) => {
        // Spawn a new Python process
        const pythonProcess = spawn(pythonExecutablePath, [pythonScriptPath, text], {
            // Set working directory to project root for relative paths in Python script
            cwd: projectRoot // Set working directory to project root for relative paths in Python script
        });

        // Initialize the result and error variables
        let result = '';
        let error = '';

        // Listen for data on the stdout 
        // stdout is for the output of the Python script
        pythonProcess.stdout.on('data', (data) => { result += data.toString(); });
        // Listen for data on the stderr stream
        // stderr is for the errors of the Python script
        pythonProcess.stderr.on('data', (data) => { error += data.toString(); });

        // Listen for the close event of the Python process
        pythonProcess.on('close', (code) => {
            // If there is an error, reject the promise with the error message
            if (error) return reject(new Error(`Python script error: ${error}`));
            // If the Python script exited with a non-zero code, reject the promise with the error message
            if (code !== 0) return reject(new Error(`Python script exited with code ${code}`));

            // Resolve the promise with the result of the Python script
            resolve(result.trim());
        });

        // Listen for the error event of the Python process
        pythonProcess.on('error', (err) => {
            // Reject the promise with the error message
            reject(new Error(`Failed to start prediction service: ${err.message}. Python: ${pythonExecutablePath}, Script: ${pythonScriptPath}`));
        });
    });
}