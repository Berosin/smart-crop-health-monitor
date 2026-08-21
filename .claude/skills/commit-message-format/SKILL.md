---
name: commit-message-format
description: Use this skill whenever the user asks for a commit message, wants to commit a change, says "commit this", asks how to describe a diff for git, or asks for a message to push changes to GitHub for this project. Always format the output using this project's specific commit message convention (imperative subject line + nested bullet body), not a generic Conventional Commits or one-line format, unless the user explicitly asks for something else.
---

# Commit Message Format (this repo)

This project uses a specific commit message convention: a short imperative
subject line, followed by a nested bullet-point body that documents *what*
changed per file/area and *why*, plus a closing verification note. Match
this shape exactly — it is what the user has asked for, not a generic
"feat:/fix:" Conventional Commits style.

## Structure

```
<Subject line: imperative, present tense, no trailing period, ~50-72 chars>

- <Area/file #1 changed>:
  * <specific function/method/change 1, plain description of what it does>
  * <specific function/method/change 2>
  * <specific function/method/change 3>
- <Area/file #2 changed, one-liner if it doesn't need sub-bullets>
- <Area/file #3 changed>:
  * <detail>
  * <detail>
- Verified: <what was checked / tested to confirm it works>
```

Rules:
- **Subject line**: imperative mood ("Replace", "Add", "Fix", "Update" — not
  "Replaced" or "Adds"). Summarize the overall change in one line, no period
  at the end.
- **Top-level bullets** (`- `): one per file or logical area touched. If the
  area has multiple distinct sub-changes worth calling out, end the bullet
  line with `:` and follow with indented sub-bullets. If it's a single,
  simple change, keep it as one bullet with no sub-bullets.
- **Sub-bullets** (two-space indent + `* `): name the specific
  function/method/component and describe concretely what it does or what
  changed about it — not vague restatements of the subject line.
- **Closing line(s)**: end with a `- Verified: ...` bullet (or
  `- Known limitation: ...` / `- Follow-up: ...` if relevant) summarizing
  what was checked, run, or confirmed working. Omit only if nothing was
  actually verified.
- Keep the tone factual and specific — reference real function names, file
  paths, and config keys from the diff, not generic descriptions.
- Do not add a "Co-authored-by" trailer or emoji unless the user asks.

## Example

**Input (what changed):** Redesigned the Streamlit UI's light theme —
new color tokens, Google Fonts (Fraunces/Inter/IBM Plex Mono), a shared
Plotly chart theme, and a `.streamlit/config.toml` for native widgets.

**Output:**

```
Redesign light theme with field-journal/spectral-scan palette

- Update utils/ui.py: new design system in inject_custom_css():
  * CSS custom properties for canopy green, soil/clay, and brick-red severity tokens
  * Google Fonts import: Fraunces (headings), Inter (body), IBM Plex Mono (scores)
  * Signature spectral gradient bar added to .pg-header and .health-card
  * CHART_THEME dict added for consistent Plotly styling across pages
- Update config.py, app.py, pages/*.py: swap ~120 hardcoded legacy hex
  values to the new palette via a consistent semantic mapping
- Add .streamlit/config.toml: theme block so native buttons/sliders/
  selects match the new primary/background/text colors
- Verified: app boots cleanly with `streamlit run app.py`, no console
  errors, all pages render with the new palette
```

## When asked for a commit message mid-conversation

If the user says something like "give me the commit message for that" right
after a change was made in the conversation, generate it from the actual
diff/edits just made (real file paths, real function names) — don't ask
them to describe the change first unless the diff isn't visible to you.
