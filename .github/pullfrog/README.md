# Pullfrog Agent User Manual (Local Pullfrog Integration)

Pullfrog is an autonomous AI coding agent integrated into Nekomimi-Waifu-Seeker CI. It runs
fully locally on GitHub Actions, enabling automated code reviews, interactive
bug-fixing, automated issue planning, and interactive task execution directly
from GitHub.

---

## How to Interact with Pullfrog

### 1. Triggering via GitHub Comments
You can invoke the agent by typing `@pullfrog` in an issue comment or pull request comment. Mentions in an **issue description** only run when the issue author is `OWNER`, `MEMBER`, or `COLLABORATOR` — the same trusted-actor rule as comment mentions. External / `FIRST_TIMER` authors do not trigger an agent run.

> [!NOTE]
> Pullfrog is configured to only listen to **explicit mentions** from trusted actors. It does not trigger automatically on standard GitHub review submission comments to avoid noisy concurrency conflict.

#### Special Command Conventions:
*   **`@pullfrog fix all`**: Directs the agent to scan, implement, and push fixes for all unresolved review threads on the current pull request.
*   **`@pullfrog fix thumbs`**: Directs the agent to address only those PR review threads that have been given a 👍 reaction by a reviewer.
*   **`@pullfrog [any custom instruction]`**: Tell the agent to execute any arbitrary coding, refactoring, or planning task.

### 2. Manual Dispatch (workflow_dispatch)
You can trigger a Pullfrog run manually through the GitHub Actions tab by selecting the **Pullfrog** workflow.

The manual dispatch accepts the following input parameters:
*   **`prompt`** (Required): The instruction or task payload for the agent.
*   **`name`** (Optional): A custom run name to identify this execution in GitHub Actions logs.
*   **`model`** (Optional): Model slug override (e.g. `anthropic/claude-3-5-sonnet`, `google/gemini-2.5-pro`). If left blank, it uses the repository variable `PULLFROG_MODEL`, falling back to `model` in `config.yml` (`opencode/muse-spark-1.3-contributor-free`).
*   **`effort`** (Optional): Claude reasoning/thinking budget override (`low` | `medium` | `high` | `max`). If left blank, it uses the repository variable `PULLFROG_EFFORT`, falling back to `medium`.
*   **`timeout`** (Optional): Maximum run duration (e.g., `20m`, `1h`). Default: `1h`.

---

## Automated Triggers & Event Workflows

Pullfrog listens to various GitHub repository events to run specific agents asynchronously. Each automated feature runs in its own isolated concurrency group to prevent jobs from cancelling each other.

| Feature / Workflow | Event Trigger | Execution Mode & Behavior | Concurrency Group |
| :--- | :--- | :--- | :--- |
| **PR Review**<br>[`pullfrog-review.yml`](../workflows/pullfrog-review.yml) | PR `opened`, `ready_for_review`, `reopened`, or `synchronize` (the last is skipped unless `on_synchronize: true`). | Uses low/high/max review budgets (default: max), tracing affected dependencies and consumers. Submits a single review, then terminates. Drafts skipped unless `include_drafts: true`. | `pullfrog-review-<repo>-<pr#>` |
| **Address Reviews**<br>[`pullfrog-address-reviews.yml`](../workflows/pullfrog-address-reviews.yml) | PR review `submitted` with `changes_requested`. | Triggers only if the reviewer is a **bot**. Pullfrog will implement the feedback, commit, and push updates. | `pullfrog-address-<repo>-<pr#>` |
| **CI Failure Fix**<br>[`pullfrog-ci-fix.yml`](../workflows/pullfrog-ci-fix.yml) | A check suite finishes with a `failure` status. | Active only on bot-authored PR branches. Pullfrog inspects logs, diagnoses/resolves the issue, and pushes a fix commit containing `[pullfrog-ci-fix]`. | `pullfrog-ci-fix-<repo>-<pr#\|sha>` |
| **Issue Triage**<br>[`pullfrog-issues.yml`](../workflows/pullfrog-issues.yml) | An issue is `opened` by `OWNER`/`MEMBER`/`COLLABORATOR`. | Pullfrog analyzes the request, writes an implementation plan as a comment, and applies up to 3 matching repository labels. Untrusted authors never start a run. | `pullfrog-issues-<repo>-<issue#>` |

---

## Configuration & Customization

### The Central Config File: [`config.yml`](config.yml)
You can configure global rules and feature flags in `.github/pullfrog/config.yml`.

```yaml
mention: "@pullfrog"

action:
  uses: CooLguNxDD/pullfrog-lemon-custom-build@<40-char SHA>  # pinned runner revision

features:
  dispatch: true             # Allow manual dispatch
  mention_triggers: true     # Respond to @pullfrog mentions in comments
  pr_review:
    enabled: true            # Enable automatic PR reviews
    budget: max              # low | high | max; PULLFROG_REVIEW_BUDGET overrides
    include_drafts: false    # Skip reviewing draft PRs
    on_synchronize: false    # Avoid expensive re-reviews on every commit push
  issues:
    enabled: true
    mode: plan               # 'plan' for plans in comments, 'build' to attempt immediate PRs
  address_reviews:
    enabled: true
    only_bot_prs: true       # Only auto-fix bot-authored PRs after a bot review
  ci_fix:
    enabled: true
    only_bot_prs: true       # Security: only auto-fix CI for bot-created PRs (avoids unchecked human writes)

status_checks: false         # Set to true to require pullfrog status checks on branch protection
```

### Agent Instruction Manuals
Under the [`instructions/`](instructions/) directory, three markdown instructions govern the agent's behavior:

1.  **[`plan.md`](instructions/plan.md)**: Governs issue enrichment. Instructs the agent to post a structured approach (goal, files affected, risks, test plan).
2.  **[`build.md`](instructions/build.md)**: Governs task execution and coding. Defines Nekomimi Python + SolidJS structure, offline testing commands, docstring requirements, and safety rules.
3.  **[`review.md`](instructions/review.md)**: Reusable code review template with low/high/max investigation budgets, evidence requirements, repository risk checks (Laya non-autoregressive boundaries, whole-word regex, candidate identity, offline tests, scraped HTML safety), and a coverage/validation summary. Max is the default.

### Code Review Budgets

Set `features.pr_review.budget` in [`config.yml`](config.yml) to `low`, `high`, or
`max` (the shipped default). The repository Actions variable
`PULLFROG_REVIEW_BUDGET` overrides this setting. Resolution order is **repository
variable → config.yml → max**; an invalid selected value falls back to `max`.
The shared workflow passes the result to automatic reviews, mention-triggered
reviews, and manual review tasks. To control the budget from config.yml, leave
the repository variable unset (remove it if it was previously configured).

| Mode | Investigation calls after checkout | Depth | Inline cap / body target |
| :--- | ---: | :--- | :--- |
| `low` | 40 | Diff, risky functions, direct callers/callees, focused checks | 10 / 400 words |
| `high` | 100 | Every changed file, affected consumers/providers, subsystem checks | 20 / 800 words |
| `max` (default) | 200 | Transitive dependency traversal, end-to-end impact, regression checks, adversarial challenge | 30 / 1,200 words |
