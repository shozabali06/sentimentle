import { useState, useEffect, useMemo } from "react";
import { Typewriter, Cursor } from "react-simple-typewriter";

// Interface for the sentiment result
export interface SentimentResult {
  text: string;
  sentiment: "positive" | "negative" | "neutral";
  confidence: number;
}

// Interface for the props
interface SentimentOutputV3Props {
  result: SentimentResult;
}

function SentimentOutputV3({ result }: SentimentOutputV3Props) {
  // State for the active line index
  const [activeLineIndex, setActiveLineIndex] = useState(0);
  // State for the completed lines
  const [completedLines, setCompletedLines] = useState<string[]>([]);
  // State for the completion status
  const [isComplete, setIsComplete] = useState(false);

  const isPositive = result.sentiment === "positive";
  const isNeutral = result.sentiment === "neutral";
  const statusColor = isNeutral
    ? "text-amber-500 dark:text-amber-400"
    : isPositive
      ? "text-green-500 dark:text-green-400"
      : "text-red-500 dark:text-red-400";
  const moodEmoji = isNeutral ? ":|" : isPositive ? ":)" : ":(";
  const cursorColor = statusColor;
  const confidencePct = Math.round(result.confidence * 100);

  // Lines to be displayed
  const lines = useMemo(() => [
    "> initializing sentiment_module... OK",
    "> parsing input... DONE",
    "> calculating score...",
    `[STATUS]: ${result.sentiment.toUpperCase()}`,
    `[CONFIDENCE]: ${confidencePct}%`,
    `[MOOD]:   ${moodEmoji}`,
  ], [result.sentiment, result.confidence]);

  const typeSpeed = 50;

  useEffect(() => {
    // Reset state when result changes
    setActiveLineIndex(0);
    setCompletedLines([]);
    setIsComplete(false);
  }, [result]);

  useEffect(() => {
    if (isComplete || activeLineIndex >= lines.length) return;

    // Get the current line to be displayed
    const currentLine = lines[activeLineIndex];
    // Calculate time needed to type the current line
    const typingTime = currentLine.length * typeSpeed;
    
    // Set timeout to move to next line after typing completes
    const timer = setTimeout(() => {
      // If not the last line, add the current line to the completed lines and move to the next line
      if (activeLineIndex < lines.length - 1) {
        setCompletedLines((prev) => [...prev, currentLine]);
        setActiveLineIndex((prev) => prev + 1);
      } else {
        // If the last line is completed, add the current line to the completed lines and set the completion status to true
        setCompletedLines((prev) => [...prev, currentLine]);
        setIsComplete(true);
      }
    }, typingTime + 100); // Small buffer after typing

    return () => clearTimeout(timer);
  }, [activeLineIndex, isComplete, lines]);

  // Get the color of the line based on the line type
  const getLineColor = (line: string) => {
    const isStatusLine = line.startsWith("[STATUS]:");
    const isConfidenceLine = line.startsWith("[CONFIDENCE]:");
    const isMoodLine = line.startsWith("[MOOD]:");
    return isStatusLine || isConfidenceLine || isMoodLine ? statusColor : "text-muted-foreground";
  };

  return (
    <div className="w-full bg-card border border-border/50 rounded px-4 py-3 font-geist-mono text-sm">
      <div className="space-y-0.5">
        {/* Display completed lines */}
        {completedLines.map((line, index) => {
          const isMoodLine = line.startsWith("[MOOD]:");
          // If the line is a mood line, add the mood emoji and the cursor
          if (isMoodLine) {
            // Split the MOOD line to add Cursor after moodEmoji
            const moodPrefix = "[MOOD]:   ";
            return (
              <div key={index} className={getLineColor(line)}>
                {moodPrefix}{moodEmoji}
                <Cursor cursorColor={cursorColor} cursorBlinking cursorStyle="█" />
              </div>
            );
          }
          // If the line is not a mood line, just return the line
          return (
            <div key={index} className={getLineColor(line)}>
              {line}
            </div>
          );
        })}

        {/* Current line being typed */}
        {!isComplete && activeLineIndex < lines.length && (
          <div className={getLineColor(lines[activeLineIndex])}>
            <Typewriter
              key={activeLineIndex}
              words={[lines[activeLineIndex]]}
              typeSpeed={typeSpeed}
              deleteSpeed={0}
              delaySpeed={0}
              loop={false}
              cursor
              cursorStyle="█"
              cursorBlinking
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default SentimentOutputV3;