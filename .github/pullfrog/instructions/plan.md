# Plan instructions (Nekomimi-Waifu-Seeker)

- Post a structured plan as an issue comment: goal, technical approach, affected files, risks, and offline test plan.
- Respect project boundaries: Laya typed decision head vs candidate sources (Playwright, Wikipedia, DDG, Gemini) vs SolidJS WebUI vs offline pytest test harness.
- Confirm docstring coverage in the plan for any new or modified functions.
- Ensure test strategy uses offline fixtures only — no live network calls or model downloads.
- Ask clarifying questions only when blocked.
- Do not open a PR unless the user (or issue body) explicitly asks to implement.
