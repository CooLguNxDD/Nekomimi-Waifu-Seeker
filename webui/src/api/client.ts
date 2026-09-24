import { bus } from "@/events/bus";

interface ErrorBody {
  error?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (res.status === 401) {
    bus.emit("session:expired", { sessionId: "" });
    throw new Error("Session expired");
  }

  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

  const data = (await res.json()) as T & ErrorBody;
  if (data && typeof data === "object" && typeof data.error === "string" && data.error) {
    if (/expired|not found/i.test(data.error)) {
      bus.emit("session:expired", { sessionId: "" });
    }
    throw new Error(data.error);
  }
  return data;
}

export const api = {
  get: <T>(p: string) => request<T>(p),
  post: <T>(p: string, b: unknown) =>
    request<T>(p, { method: "POST", body: JSON.stringify(b) }),
};
