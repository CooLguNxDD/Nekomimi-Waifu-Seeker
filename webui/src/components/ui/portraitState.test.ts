import assert from "node:assert/strict";
import { portraitFailedAfterIdentityChange, portraitShowsImage } from "./portraitState.ts";

assert.equal(portraitShowsImage(null, false), false);
assert.equal(portraitShowsImage("https://cdn.example/hoshino.png", false), true);
assert.equal(portraitShowsImage("https://cdn.example/broken.png", true), false);
// The next guess must not inherit the previous onError.
assert.equal(portraitShowsImage("https://cdn.example/rossina.png", portraitFailedAfterIdentityChange()), true);

console.log("portrait state tests ok");
