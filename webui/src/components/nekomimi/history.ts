import { createEffect, createSignal, onCleanup } from "solid-js";
import { useQueryClient } from "@tanstack/solid-query";
import { fetchState } from "@/api/nekomimi";
import { nekomimiQueryKey } from "@/hooks/useNekomimi";
import type { NekomimiQuestion, NekomimiState } from "@/types/game";
import {
  appendChip,
  chipForAnswer,
  chipForRejection,
  chipsFromAsked,
  clearTranscript,
  loadTranscript,
  mergeChips,
  saveTranscript,
  type AnswerChipData,
} from "@/components/nekomimi/transcript";

export {
  chipText,
  chipsFromAsked,
  displayLabel,
  type AnswerChipData,
} from "@/components/nekomimi/transcript";

/**
 * Own the answer trail for the open round.
 *
 * Turn responses omit ``asked``. A remount must not treat that cache as the
 * transcript: it fetches the snapshot first, then merges it with chips already
 * committed after a successful answer.
 */
export function useAnswerTranscript(
  state: () => NekomimiState | undefined,
  sessionId: () => string | null,
) {
  const qc = useQueryClient();
  const [chips, setChips] = createSignal<AnswerChipData[]>(loadTranscript(sessionId() ?? ""));
  const [seen, setSeen] = createSignal<string | null>(null);

  const write = (sid: string, next: AnswerChipData[]) => {
    saveTranscript(sid, next);
    setChips(next);
  };

  createEffect(() => {
    const current = state();
    const sid = current?.session_id;
    if (!sid || sid === seen()) return;
    if (Array.isArray(current.asked)) {
      write(sid, mergeChips(chipsFromAsked(current.seed, current.asked), loadTranscript(sid)));
      setSeen(sid);
      return;
    }
    let cancelled = false;
    onCleanup(() => {
      cancelled = true;
    });
    fetchState(sid)
      .then((snap) => {
        if (cancelled) return;
        qc.setQueryData(nekomimiQueryKey(sid), snap);
        write(sid, mergeChips(chipsFromAsked(snap.seed, snap.asked), loadTranscript(sid)));
        setSeen(sid);
      })
      .catch(() => {
        // Leave the session unseen so the next state change tries the snapshot again.
      });
  });

  return {
    chips,
    /** Record an answer only after the server accepted it. */
    commitAnswer(
      sid: string,
      question: NekomimiQuestion | undefined,
      answer: string,
      detail?: string,
      label?: string,
    ) {
      write(sid, appendChip(loadTranscript(sid), chipForAnswer(question, answer, detail, label)));
    },
    /** Record a rejected guess only after the server accepted the "no". */
    commitRejection(sid: string, name: string, candidateId?: string) {
      write(sid, appendChip(loadTranscript(sid), chipForRejection(name, candidateId)));
    },
    reset(sid: string) {
      clearTranscript(sid);
      setChips([]);
      setSeen(null);
    },
  };
}
