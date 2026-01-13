import { useState, useEffect, useMemo } from "react";
import { Typewriter, Cursor } from "react-simple-typewriter";

export interface SentimentResult {
  text: string;
  sentiment: "positive" | "negative";
}

interface SentimentOutputV3Props {
  result: SentimentResult;
}

function SentimentOutputV3({ result }: SentimentOutputV3Props) {
  const [activeLineIndex, setActiveLineIndex] = useState(0);
  const [completedLines, setCompletedLines] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  const isPositive = result.sentiment === "positive";
  const statusColor = isPositive ? "text-green-500 dark:text-green-400" : "text-red-500 dark:text-red-400";
  const moodEmoji = isPositive ? ":)" : ":(";
  const cursorColor = statusColor;

  const lines = useMemo(() => [
    "> initializing sentiment_module... OK",
    "> parsing input... DONE",
    "> calculating score...",
    `[STATUS]: ${result.sentiment.toUpperCase()}`,
    `[MOOD]:   ${moodEmoji}`,
  ], [result.sentiment]);

  const typeSpeed = 50;

  useEffect(() => {
    // Reset state when result changes
    setActiveLineIndex(0);
    setCompletedLines([]);
    setIsComplete(false);
  }, [result]);

  useEffect(() => {
    if (isComplete || activeLineIndex >= lines.length) return;

    const currentLine = lines[activeLineIndex];
    // Calculate time needed to type the current line
    const typingTime = currentLine.length * typeSpeed;
    
    // Set timeout to move to next line after typing completes
    const timer = setTimeout(() => {
      if (activeLineIndex < lines.length - 1) {
        setCompletedLines((prev) => [...prev, currentLine]);
        setActiveLineIndex((prev) => prev + 1);
      } else {
        // Last line completed
        setCompletedLines((prev) => [...prev, currentLine]);
        setIsComplete(true);
      }
    }, typingTime + 100); // Small buffer after typing

    return () => clearTimeout(timer);
  }, [activeLineIndex, isComplete, lines]);

  const getLineColor = (line: string) => {
    const isStatusLine = line.startsWith("[STATUS]:");
    const isMoodLine = line.startsWith("[MOOD]:");
    return isStatusLine || isMoodLine ? statusColor : "text-muted-foreground";
  };

  return (
    <div className="w-full bg-card border border-border/50 rounded px-4 py-3 font-geist-mono text-sm">
      <div className="space-y-0.5">
        {/* Display completed lines */}
        {completedLines.map((line, index) => {
          const isMoodLine = line.startsWith("[MOOD]:");
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