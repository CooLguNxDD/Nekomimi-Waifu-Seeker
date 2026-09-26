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
  isTranscriptRetired,
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

  /** Remember ``sid`` and show it only while that round is still the one on screen. */
  const write = (sid: string, next: AnswerChipData[]) => {
    if (isTranscriptRetired(sid)) return;
    saveTranscript(sid, next);
    if ((state()?.session_id ?? sessionId() ?? "") !== sid) return;
    setChips(next);
  };

  createEffect(() => {
    const current = state();
    const sid = current?.session_id;
    if (!sid || sid === seen() || isTranscriptRetired(sid)) return;
    if (Array.isArray(current.asked)) {
      write(sid, mergeChips(chipsFromAsked(current.seed, current.asked), loadTranscript(sid)));
      setSeen(sid);
      return;
    }
    // Turn payloads omit ``asked``. Paint the local trail now so Restart does
    // not keep the previous round's chips until this snapshot returns.
    setChips(loadTranscript(sid));
    let cancelled = false;
    onCleanup(() => {
      cancelled = true;
    });
    fetchState(sid)
      .then((snap) => {
        if (cancelled || isTranscriptRetired(sid)) return;
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
    /** Hide the trail without retiring it, so a failed restart can fetch it back. */
    blank(sid: string) {
      setChips([]);
      setSeen(sid || null);
    },
    /** Let the loader fetch ``sid`` again after a restart that never left this round. */
    reopen(sid: string) {
      if (sid && seen() === sid) setSeen(null);
    },
    /** Drop the trail for good. A round already on screen keeps the chips it loaded. */
    reset(sid: string) {
      clearTranscript(sid);
      if ((state()?.session_id ?? sessionId() ?? "") !== sid) return;
      setChips([]);
      setSeen(sid || null);
    },
  };
}
