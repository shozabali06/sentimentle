import SentimentAnalyzer from "./components/SentimentAnalyzer";
import { ThemeProvider } from "./components/theme-provider";

function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <SentimentAnalyzer />
    </ThemeProvider>
  );
}

export default App;
