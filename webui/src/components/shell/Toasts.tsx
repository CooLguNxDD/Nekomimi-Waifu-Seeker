import { createSignal, For, onCleanup, onMount } from "solid-js";
import { bus } from "@/events/bus";
import { sessionStore } from "@/store";

/** Toast list fed by the mitt bus. Session expiry clears the stored round. */
export function Toasts() {
  const [messages, setMessages] = createSignal<string[]>([]);

  onMount(() => {
    const onToast = (event: { message: string }) => {
      setMessages((prev) => [...prev, event.message].slice(-3));
    };
    const onExpired = () => {
      sessionStore.getState().clearSession();
      onToast({ message: "That round expired. Start a new one." });
    };
    bus.on("toast", onToast);
    bus.on("session:expired", onExpired);
    onCleanup(() => {
      bus.off("toast", onToast);
      bus.off("session:expired", onExpired);
    });
  });

  return (
    <div class="fixed bottom-4 right-4 flex max-w-sm flex-col gap-2">
      <For each={messages()}>
        {(message) => (
          <div class="rounded-[10px] border border-border bg-surface px-3 py-2 text-sm text-danger">
            {message}
          </div>
        )}
      </For>
    </div>
  );
}
