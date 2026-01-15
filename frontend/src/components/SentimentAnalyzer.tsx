import { Textarea } from "@/components/ui/textarea";
import { Button } from "./ui/button";
import { type SentimentResult } from "./SentimentOutputV3";
import { ModeToggle } from "./mode-toggle";
import { useState, useEffect } from "react";
import SentimentOutputV3 from "./SentimentOutputV3";

function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const totalChars = 500;
  const charCount = text.length;
  const isOverLimit = charCount > totalChars;
  const isDisabled = charCount === 0 || isOverLimit;

  const analyzeSentiment = async () => {
    if (isDisabled) return;

    setResult(null);
    setIsAnalyzing(true);
    setError("");

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

    try {
      const response = await fetch(`${API_URL}/api/analyze-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
      setError("");
    } catch (error) {
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("An unknown error occurred.");
      }
    } finally {
      setIsAnalyzing(false);
      setError("");
    }
  };

  // Handle Enter key to analyze sentiment
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      analyzeSentiment();
    }
  };

  // Clear the result and reset the textarea
  const handleClear = () => {
    setResult(null);
    setText("");
    setError("");
  };

  // Warn user before reloading if textarea has content
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (text.trim().length > 0) {
        e.preventDefault();
        return "";
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [text]);

  return (
    <div className="h-dvh bg-background text-foreground font-geist-mono relative">
      {/* Header */}
      <header className="border-b border-border/50 px-4 py-6 sm:px-6">
        <div className="max-w-2xl mx-auto flex items-start justify-between">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-balance">
              sentimentle
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              analyze the sentiment of your text
            </p>
          </div>
          <ModeToggle />
        </div>
      </header>

      {/* Main Content */}
      <main className="px-4 py-8 sm:px-6 sm:py-12">
        <div className="max-w-2xl mx-auto space-y-3">
          {/* Input Textarea */}
          <div className="space-y-3">
            <div className="flex items-baseline justify-between">
              <label
                htmlFor="text-input"
                className="text-sm font-medium text-foreground"
              >
                enter text
              </label>
              <span className="text-xs text-muted-foreground">
                {charCount}/{totalChars}
              </span>
            </div>
          </div>
          <Textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value.slice(0, totalChars));
              setError("");
            }}
            onKeyDown={handleKeyDown}
            id="text-input"
            className="w-full min-h-32 max-h-96 bg-card border border-border/50 rounded px-4 py-3 text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-colors font-geist-mono resize-none"
            placeholder="type something to analyze"
            disabled={isAnalyzing}
          />
          {isOverLimit && (
            <p className="text-xs text-red-500 font-medium">
              ✗ text exceeds {totalChars} character limit
            </p>
          )}
          {error && (
            <p className="text-xs text-red-500 font-medium">
              ✗ {error}
            </p>
          )}

          {/* Action buttons */}
          <div className="flex gap-2">
            <Button
              onClick={analyzeSentiment}
              disabled={isDisabled || isAnalyzing}
              className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm rounded-sm"
            >
              {isAnalyzing ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="inline-block w-1.5 h-1.5 bg-current rounded-full animate-pulse"></span>
                  analyzing
                </span>
              ) : (
                "analyze"
              )}
            </Button>
            {result && (
              <Button
                onClick={handleClear}
                variant="outline"
                className="px-4 border-border/50 text-foreground hover:bg-card text-sm bg-transparent rounded-sm"
              >
                clear
              </Button>
            )}
          </div>

          {/* result */}
          <div className="space-y-4">
            {result && <SentimentOutputV3 result={result} />}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="absolute bottom-5 w-full">
        <div className="max-w-2xl mx-auto text-center">
          <p className="text-xs text-muted-foreground">
            Made with ❤️ by{" "}
            <a
              href="https://github.com/shozabali06"
              target="_blank"
              rel="noopener noreferrer"
              className="text-foreground hover:text-primary transition-colors underline underline-offset-4"
            >
              Shozab
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default SentimentAnalyzer;
