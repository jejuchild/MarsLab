import { useState, useCallback, useEffect, useMemo } from "react";
import toast from "react-hot-toast";
import { listFieldNotes } from "../api/fieldnotes";
import type { FieldNote } from "../api/fieldnotes";

const EMPTY_NOTES: FieldNote[] = [];

export default function useFieldNotes() {
  const [fieldNotes, setFieldNotes] = useState<FieldNote[]>([]);
  const [showFieldNotesOnMap, setShowFieldNotesOnMap] = useState(false);
  const [fieldNoteActiveTag, setFieldNoteActiveTag] = useState<string | null>(null);
  const [showFieldNoteModal, setShowFieldNoteModal] = useState<{
    productId: string; instrument: string; lat: number; lon: number;
  } | null>(null);

  // Load on mount
  useEffect(() => {
    listFieldNotes().then(setFieldNotes).catch(console.error);
  }, []);

  const refreshFieldNotes = useCallback(() => {
    listFieldNotes().then((notes) => {
      setFieldNotes(notes);
      toast.success("Field note saved");
    }).catch(console.error);
  }, []);

  const handleOpenFieldNote = useCallback((productId: string, instrument: string, lat: number, lon: number) => {
    setShowFieldNoteModal({ productId, instrument, lat, lon });
  }, []);

  // Notes to show on map: only those matching the active tag
  const mapFieldNotes = useMemo(
    () => fieldNoteActiveTag
      ? fieldNotes.filter(n => n.tags.includes(fieldNoteActiveTag))
      : [],
    [fieldNotes, fieldNoteActiveTag]
  );

  const mapFieldNotesForView = useMemo(
    () => showFieldNotesOnMap ? mapFieldNotes : EMPTY_NOTES,
    [showFieldNotesOnMap, mapFieldNotes]
  );

  return {
    fieldNotes,
    showFieldNoteModal,
    setShowFieldNoteModal,
    showFieldNotesOnMap,
    setShowFieldNotesOnMap,
    fieldNoteActiveTag,
    setFieldNoteActiveTag,
    mapFieldNotesForView,
    refreshFieldNotes,
    handleOpenFieldNote,
  };
}
