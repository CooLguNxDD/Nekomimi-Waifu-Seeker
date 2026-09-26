import assert from "node:assert/strict";
import { questionHeading } from "./questionHeading.ts";

const hair = questionHeading({
  text: "What colour is your character's hair?",
  kind: "choice",
});
assert.equal(hair, "What colour is your character's hair?");

assert.equal(
  questionHeading({ text: "  Which series is your character from?  ", kind: "choice" }),
  "Which series is your character from?",
);

assert.equal(questionHeading({ text: "", kind: "choice" }), "Pick the closest option.");
assert.equal(questionHeading({ text: "   ", kind: "yesno" }), "Does this describe your character?");
assert.equal(questionHeading(undefined), "Does this describe your character?");

console.log("question heading tests ok");
