import app from './app.js';

// Get the port from the environment variables or use 3000 as default
const PORT = process.env.PORT || 3000;

// Start the server on the specified port
app.listen(Number(PORT), () => {
  // Log the server running message with the port
  console.log(`Server running on port ${PORT}`);
});