"use client";

import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export const MAX_QUESTION_CHARS = 2000;

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
}

export function ChatInput({ value, onChange, onSubmit, onStop, busy, disabled }: ChatInputProps) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const tooLong = value.length > MAX_QUESTION_CHARS;
  const canSend = !busy && !disabled && value.trim().length > 0 && !tooLong;

  useEffect(() => {
    if (!busy) ref.current?.focus();
  }, [busy]);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter submits; Shift+Enter inserts a newline; ignore while composing (IME).
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (canSend) onSubmit();
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (canSend) onSubmit();
  };

  return (
    <form onSubmit={handleSubmit} className="relative rounded-xl border bg-background shadow-xs focus-within:ring-3 focus-within:ring-ring/30">
      <label htmlFor="chat-input" className="sr-only">
        Ask a question about company policies
      </label>
      <Textarea
        id="chat-input"
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about a policy… (Enter to send, Shift+Enter for a new line)"
        rows={1} // height grows with content via CSS field-sizing (capped by max-h-56)
        disabled={disabled}
        aria-invalid={tooLong}
        className="max-h-56 min-h-12 resize-none border-0 bg-transparent py-3 pr-14 pl-4 shadow-none focus-visible:ring-0 dark:bg-transparent"
      />
      <div className="absolute right-2 bottom-2 flex items-center gap-2">
        {value.length > MAX_QUESTION_CHARS * 0.8 && (
          <span className={tooLong ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
            {value.length}/{MAX_QUESTION_CHARS}
          </span>
        )}
        {busy ? (
          <Button type="button" size="icon-sm" variant="secondary" onClick={onStop} aria-label="Stop generating">
            <Square className="fill-current" />
          </Button>
        ) : (
          <Button type="submit" size="icon-sm" disabled={!canSend} aria-label="Send question">
            <ArrowUp />
          </Button>
        )}
      </div>
    </form>
  );
}
