import mitt from "mitt";

export type AppEvents = {
  "toast": { message: string };
  "session:expired": { sessionId: string };
  "guess:resolved": { correct: boolean };
};

export const bus = mitt<AppEvents>();
