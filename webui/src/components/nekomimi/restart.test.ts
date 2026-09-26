import assert from "node:assert/strict";
import { restartNeedsConfirm } from "./restart.ts";

assert.equal(restartNeedsConfirm("asking"), true);
assert.equal(restartNeedsConfirm("guessing"), true);
assert.equal(restartNeedsConfirm("done"), false);
assert.equal(restartNeedsConfirm(undefined), false);

console.log("restart confirm tests ok");
