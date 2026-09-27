# Grok and Antigravity Pipeline Setup Guide

This guide describes how to configure and use the **Grok** (xAI) and **Antigravity** (Google) agents within the local Pullfrog pipeline of this repository.

---

## Pipeline Configurations

The pipeline uses the custom Pullfrog Action runner and agent image:
*   **Custom Runner Action**: pinned SHA in `.github/pullfrog/config.yml` (`action.uses`)
*   **Docker Agent Image**: digest-pinned in `.github/pullfrog/config.yml` (`image.digest`)

---

## GitHub Secrets Configuration

To run these agents, you must configure their respective credentials in your repository's **Settings → Secrets and variables → Actions**.

### 1. Google Antigravity Setup
*   **Secret Name**: `ANTIGRAVITY_TOKEN`
*   **Purpose**: Authenticates the Antigravity CLI (`agy`) inside the pipeline container.
*   **Configuration**:
    1. Retrieve your Antigravity token (e.g., from your local configuration or the Antigravity console).
    2. Add the token value as a GitHub repository secret named `ANTIGRAVITY_TOKEN`.

### 2. xAI Grok Setup
*   **Secret Name**: `GROK_AUTH_JSON`
*   **Purpose**: Authenticates the Grok Build CLI (`grok`) inside the pipeline container.
*   **Configuration**:
    1. Locate your local Grok credentials file at `~/.grok/auth.json`.
    2. Copy the **raw JSON string** (or encode it in Base64 if preferred, as both formats are supported).
    3. Add the JSON/Base64 string as a GitHub repository secret named `GROK_AUTH_JSON`.

---

## Running the Agents on the Pipeline

When triggering the agent manually or programmatically, you can specify the model override options:

*   **Antigravity Model ID**: For Google Antigravity runs, configure your environment or override the model parameter accordingly (defaulting to the specified Antigravity model configuration).
*   **Grok Model ID**: Set the model input to `xai/grok-beta` (or other supported xAI models) to dispatch runs to the Grok harness.
