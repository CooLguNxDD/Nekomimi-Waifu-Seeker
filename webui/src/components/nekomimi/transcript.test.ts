import assert from "node:assert/strict";
import {
  appendChip,
  chipForAnswer,
  chipsFromAsked,
  displayLabel,
  mergeChips,
} from "./transcript.ts";

const choice = chipsFromAsked("silver hair", [
  {
    qid: "series",
    text: "Which series is your character from?",
    answer: "eva",
    label: "Neon Genesis Evangelion",
  },
  { qid: "hair", text: "Does your character have red hair?", answer: "yes" },
  { qid: "pending", text: "Next?", answer: null },
]);

assert.equal(choice[0]?.label, "silver hair");
assert.equal(choice[1]?.label, "Neon Genesis Evangelion");
assert.equal(choice[2]?.label, "Yes");
assert.equal(choice.length, 3);
assert.equal(displayLabel("detail", "she pilots a mech"), "she pilots a mech");

const committed = appendChip(
  [],
  chipForAnswer({ qid: "series", text: "Which series?" }, "eva", undefined, "Neon Genesis Evangelion"),
);
assert.deepEqual(appendChip(committed, committed[0]!), committed);

const merged = mergeChips(choice, [
  ...committed,
  { id: "reject:asuka", label: "Not Asuka", question: "Rejected guess" },
]);
assert.equal(merged.at(-1)?.label, "Not Asuka");
assert.equal(merged.filter((chip) => chip.id === "series").length, 1);

console.log("transcript tests ok");
