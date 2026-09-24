import { createSignal, For, onCleanup, onMount } from "solid-js";
import { bus } from "@/events/bus";
import { sessionStore } from "@/store";

type Toast = { id: number; message: string };

/** Toast list fed by the mitt bus. Each toast leaves on its own timer. */
export function Toasts() {
  const [messages, setMessages] = createSignal<Toast[]>([]);
  let nextId = 0;

  onMount(() => {
    const timers: number[] = [];
    const onToast = (event: { message: string }) => {
      const id = ++nextId;
      setMessages((prev) => [...prev, { id, message: event.message }].slice(-3));
      timers.push(
        window.setTimeout(() => {
          setMessages((prev) => prev.filter((item) => item.id !== id));
        }, 5000),
      );
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
      timers.forEach((timer) => window.clearTimeout(timer));
    });
  });

  return (
    <div class="fixed bottom-4 right-4 flex max-w-sm flex-col gap-2">
      <For each={messages()}>
        {(toast) => (
          <div class="rounded-[10px] border border-border bg-surface px-3 py-2 text-sm text-danger">
            {toast.message}
          </div>
        )}
      </For>
    </div>
  );
}
