import { createSignal, onCleanup } from "solid-js";
import type { StoreApi } from "zustand/vanilla";

/** Subscribe a Solid accessor to a vanilla Zustand store. */
export function useStore<T, U>(store: StoreApi<T>, selector: (state: T) => U): () => U {
  const [value, setValue] = createSignal(selector(store.getState()));
  const unsubscribe = store.subscribe((state) => {
    const next = selector(state);
    setValue(() => next);
  });
  onCleanup(unsubscribe);
  return value;
}
