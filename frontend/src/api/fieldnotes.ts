/* Field Notes API client */

export interface FieldNote {
  id: string;
  product_id: string;
  instrument: string;
  lat: number;
  lon: number;
  tags: string[];
  memo: string;
  created_at: string;
  updated_at: string;
}

export interface NoteCreatePayload {
  product_id: string;
  instrument: string;
  lat: number;
  lon: number;
  tags: string[];
  memo: string;
}

export interface NoteUpdatePayload {
  tags?: string[];
  memo?: string;
}

export async function listFieldNotes(): Promise<FieldNote[]> {
  const res = await fetch("/api/fieldnotes");
  if (!res.ok) throw new Error(`Failed to list notes: ${res.status}`);
  return res.json();
}

export async function createFieldNote(payload: NoteCreatePayload): Promise<FieldNote> {
  const res = await fetch("/api/fieldnotes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to create note: ${res.status}`);
  return res.json();
}

export async function updateFieldNote(noteId: string, payload: NoteUpdatePayload): Promise<FieldNote> {
  const res = await fetch(`/api/fieldnotes/${noteId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to update note: ${res.status}`);
  return res.json();
}

export async function deleteFieldNote(noteId: string): Promise<void> {
  const res = await fetch(`/api/fieldnotes/${noteId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete note: ${res.status}`);
}

export async function getNotesForProduct(productId: string): Promise<FieldNote[]> {
  const res = await fetch(`/api/fieldnotes/product/${encodeURIComponent(productId)}`);
  if (!res.ok) throw new Error(`Failed to get notes: ${res.status}`);
  return res.json();
}

export async function getAllTags(): Promise<string[]> {
  const res = await fetch("/api/fieldnotes/tags");
  if (!res.ok) throw new Error(`Failed to get tags: ${res.status}`);
  return res.json();
}
