# 🤖 Agent Playbook — nsys-ai

This document is the onboarding guide for AI agents working on `nsys-ai`.

---

## Repository Structure

```
nsys-ai/
├── src/nsys_tui/          # Main Python package
│   ├── __main__.py        # CLI entry point (click-based)
│   ├── profile.py         # SQLite profile loader
│   ├── tui.py             # Tree TUI (curses)
│   ├── tui_timeline.py    # Timeline TUI (curses)
│   ├── tree.py            # NVTX tree data model
│   ├── search.py          # Kernel/NVTX search
│   ├── overlap.py         # Compute/NCCL overlap analysis
│   ├── summary.py         # Profile summary stats
│   ├── export.py          # HTML viewer export
│   ├── export_flat.py     # CSV/JSON flat export
│   ├── projection.py      # Time-range projection
│   ├── viewer.py          # Perfetto JSON export
│   ├── web.py             # Flask web UI server
│   ├── ai/                # AI module (commentary, suggestions)
│   └── templates/         # HTML templates (Jinja2)
├── tests/                 # pytest test suite
│   └── test_cli.py        # CLI smoke tests
├── docs/                  # Documentation (8 guides)
├── site/                  # GitHub Pages landing page
├── examples/              # Example HTML exports
├── pyproject.toml         # Package config (setuptools)
├── ROADMAP.md             # Prioritized roadmap with issue links
└── .github/workflows/
    ├── ci.yml             # CI: Python 3.10/3.11/3.12
    ├── publish.yml        # PyPI publish on v* tag
    ├── pages.yml          # GitHub Pages on site/** changes
    ├── project-sync.yml   # Label → project board sync
    └── security.yml       # Security scanning
```

---

## Label System

### Priority Labels
| Label | Meaning |
|-------|---------|
| `P0-critical` | Must have — next sprint |
| `P1-high` | High priority — near term |
| `P2-medium` | Medium priority |
| `P3-low` | Nice to have — longer term |

### Pillar Labels
| Label | Meaning |
|-------|---------|
| `pillar/ai` | AI pillar (analysis, NLP, models) |
| `pillar/ui` | UI pillar (TUI, web, viewer) |

### Agent Workflow Labels (State Machine)

```
agent-ready → agent-in-progress → agent-review → (merged) → Done
                    ↕
              agent-blocked
```

| Label | Meaning | When to apply |
|-------|---------|---------------|
| `agent-ready` | Issue is fully spec'd, ready to pick up | Default for roadmap items |
| `agent-in-progress` | Agent is actively working on this | After claiming the issue |
| `agent-blocked` | Agent cannot proceed (needs info/dependency) | When stuck |
| `agent-review` | PR raised, awaiting human review | After `gh pr create` |

---

## Agent Workflow

### 1. Find an issue to work on

```bash
# List all agent-ready issues, highest priority first
gh issue list -R GindaChen/nsys-ai --label agent-ready --sort created
```

Pick the highest-priority issue (`P0` > `P1` > `P2` > `P3`).

### 2. Claim the issue

```bash
# Swap label: agent-ready → agent-in-progress
gh issue edit <NUM> -R GindaChen/nsys-ai \
  --remove-label "agent-ready" \
  --add-label "agent-in-progress"
```

### 3. Create a branch

```bash
git checkout main && git pull
git checkout -b feat/issue-<NUM>-<short-description>
```

Branch naming: `feat/issue-3-ask-command`, `fix/issue-9-multi-gpu`, etc.

### 4. Implement

- Code lives in `src/nsys_tui/`
- Add tests in `tests/`
- Follow existing code style (no formatter configured — match surrounding code)
- The package name is `nsys_tui` internally (Python module), `nsys-ai` externally (PyPI)

### 5. Test before pushing

```bash
# Install in dev mode
pip install -e '.[dev]'

# Smoke test CLI
python -m nsys_tui --help

# Run test suite
pytest tests/ -v --tb=short
```

All three must pass. CI runs on Python 3.10, 3.11, and 3.12.

### 6. Push and create PR

```bash
git push -u origin feat/issue-<NUM>-<short-description>

gh pr create -R GindaChen/nsys-ai \
  --title "<type>: <description>" \
  --body "Closes #<NUM>\n\n<summary of changes>"
```

Then update the label:

```bash
gh issue edit <NUM> -R GindaChen/nsys-ai \
  --remove-label "agent-in-progress" \
  --add-label "agent-review"
```

### 7. Wait for CI + review, then merge

```bash
# Check CI status
gh pr checks <PR_NUM> -R GindaChen/nsys-ai

# Merge when approved and green
gh pr merge <PR_NUM> -R GindaChen/nsys-ai --squash --delete-branch
```

---

## Testing Checklist

Before raising a PR, verify:

- [ ] `python -m nsys_tui --help` — CLI loads without error
- [ ] `pytest tests/ -v --tb=short` — all tests pass
- [ ] If you added a new CLI command, add a smoke test in `tests/test_cli.py`
- [ ] If you modified `site/`, check the page renders (`open site/index.html`)
- [ ] If you modified AI code, test with `pip install -e '.[ai]'`

---

## Deployment

### PyPI Release
1. Bump `version` in `pyproject.toml`
2. Commit: `git commit -m "chore: bump to vX.Y.Z"`
3. Tag + push: `git tag vX.Y.Z && git push origin main --tags`
4. The `publish.yml` workflow auto-publishes to PyPI

### GitHub Pages
- Modify files in `site/` and push to `main`
- The `pages.yml` workflow auto-deploys to https://gindachen.github.io/nsys-ai/

---

## Key Design Decisions

- **Internal module = `nsys_tui`**, external package = `nsys-ai` (historical rename)
- **No runtime dependencies** for core TUI — only stdlib (`curses`, `sqlite3`, `json`)
- **Optional deps**: `anthropic` for AI features (`pip install nsys-ai[ai]`), `pytest` for dev
- **Profiles are `.sqlite` files** exported from NVIDIA Nsight Systems (`.nsys-rep` → `.sqlite`)
- **Two TUI modes**: tree (hierarchical NVTX browser) and timeline (horizontal kernel view)

---

## Project Board

Project: [nsys-ai (#5)](https://github.com/users/GindaChen/projects/5)

The `project-sync.yml` workflow automatically moves issues between columns when agent labels change. Requires the `PROJECT_TOKEN` secret.
