import { useState, useCallback } from "react";

export type UndoAction = {
  type: string;
  description: string;  // Human-readable (e.g., "Toggle CRISM overlay")
  undo: () => void;
  redo: () => void;
};

export function useUndoRedo(maxHistory: number = 20) {
  const [undoStack, setUndoStack] = useState<UndoAction[]>([]);
  const [redoStack, setRedoStack] = useState<UndoAction[]>([]);

  const pushAction = useCallback((action: UndoAction) => {
    setUndoStack(prev => [...prev.slice(-(maxHistory - 1)), action]);
    setRedoStack([]); // Clear redo stack on new action
  }, [maxHistory]);

  const undo = useCallback(() => {
    setUndoStack(prev => {
      if (prev.length === 0) return prev;
      const action = prev[prev.length - 1];
      action.undo();
      setRedoStack(r => [...r, action]);
      return prev.slice(0, -1);
    });
  }, []);

  const redo = useCallback(() => {
    setRedoStack(prev => {
      if (prev.length === 0) return prev;
      const action = prev[prev.length - 1];
      action.redo();
      setUndoStack(u => [...u, action]);
      return prev.slice(0, -1);
    });
  }, []);

  const canUndo = undoStack.length > 0;
  const canRedo = redoStack.length > 0;
  const lastAction = undoStack[undoStack.length - 1] ?? null;

  return { pushAction, undo, redo, canUndo, canRedo, lastAction };
}
