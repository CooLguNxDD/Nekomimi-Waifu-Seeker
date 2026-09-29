# Pullfrog Environment & Secrets Setup Guide

This guide details how to configure GitHub Actions environment variables and secrets to run Pullfrog with **OpenCode** and **Muse Spark 1.3 (Free / Contributor model)** on Nekomimi-Waifu-Seeker.

---

## Quick Setup via `gh` CLI

Run these commands from your local terminal with the GitHub CLI authenticated:

```bash
# 1. Repository Variables (Configuration)
gh variable set PULLFROG_AGENT --body "opencode"
gh variable set PULLFROG_MODEL --body "opencode/muse-spark-1.3-contributor-free"
gh variable set PULLFROG_EFFORT --body "medium"

# 2. Authentication Secret (Choose Option A or Option B)
# Option A: Personal Access Token (classic PAT or fine-grained token with contents:write, pull-requests:write, issues:write)
gh secret set PULLFROG_GITHUB_TOKEN

# Option B: GitHub App credentials (recommended if using an App)
gh secret set PULLFROG_APP_ID
gh secret set PULLFROG_APP_PRIVATE_KEY < path/to/private-key.pem

# 3. Model API Key for OpenRouter (Muse Spark 1.3 Free/Contributor):
gh secret set OPENROUTER_API_KEY
```

---

## Manual Setup via GitHub Web UI

Navigate to **Settings → Secrets and variables → Actions** in your repository:

### 1. Variables Tab (`Repository variables`)
Click **New repository variable**:

| Variable Name | Value | Description |
| :--- | :--- | :--- |
| `PULLFROG_AGENT` | `opencode` | Forces the OpenCode agent harness. |
| `PULLFROG_MODEL` | `opencode/muse-spark-1.3-contributor-free` | Default model (Muse Spark 1.3 Contributor / Free tier via OpenCode). |
| `PULLFROG_EFFORT` | `medium` | Reasoning effort level (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). |

### 2. Secrets Tab (`Repository secrets`)
Click **New repository secret**:

| Secret Name | Description |
| :--- | :--- |
| `OPENROUTER_API_KEY` | **Required**: OpenRouter API key for Muse Spark 1.3. |
| `PULLFROG_GITHUB_TOKEN` | GitHub PAT with `contents: write`, `pull-requests: write`, `issues: write`. |
| `META_MODEL_API_KEY` | *(Alternative)* Direct Meta Model API key if switching to direct Meta route. |
| `OPENCODE_API_KEY` | *(Alternative)* OpenCode Zen API key if routing via OpenCode Zen. |
| `PULLFROG_APP_ID` | *(Optional)* GitHub App ID if authenticating via GitHub App. |
| `PULLFROG_APP_PRIVATE_KEY` | *(Optional)* GitHub App private RSA key PEM. |
| `ANTIGRAVITY_TOKEN` | *(Optional)* Antigravity CLI token for Google Antigravity agent runs. |
| `GROK_AUTH_JSON` | *(Optional)* Grok Build auth JSON/base64 for xAI Grok agent runs. |

---

## Model Routing Details for Muse Spark 1.3

Pullfrog supports Muse Spark 1.3 across multiple provider backends:

1. **Meta Direct Route (`meta/muse-spark-contributor` or `meta/muse-spark-1.3`)**:
   - Requires `META_MODEL_API_KEY`.
   - The Contributor variant (`meta/muse-spark-contributor`) uses Meta's discounted training-opt-in tier ($0.10/$0.20 per M tokens, cache read $0.002).
2. **OpenRouter Route (`openrouter/meta/muse-spark-1.3` or `openrouter/meta/muse-spark-1.3-contributor`)**:
   - Requires `OPENROUTER_API_KEY`.
   - Enables BYOK routing through OpenRouter.
3. **OpenCode Zen Route (`opencode/muse-spark-1.3`)**:
   - Requires `OPENCODE_API_KEY`.
