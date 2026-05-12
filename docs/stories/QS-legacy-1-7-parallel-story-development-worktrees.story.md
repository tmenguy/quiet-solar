# Story 1.7: Parallel Story Development with Git Worktrees

Status: review

## Story

As TheDev,
I want every story and bugfix to automatically use a git worktree (with shared venv and HA runtime config via symlinks) by default, with the option to opt out by saying "no worktree",
So that I can work on multiple stories in parallel without branch switching, stashing, or duplicating large dependencies, while keeping the flexibility to skip worktrees for trivial fixes.

## Acceptance Criteria

1. **Given** TheDev starts any story or bugfix (via `/bmad-dev-story` or `/bmad-quick-dev-new-preview`)
   **When** Phase 1b (branch creation) executes
   **Then** a worktree is created by default with its own branch
   **And** venv, config/, and non-git custom_components are symlinked from the main worktree
   **And** the main worktree stays on main, untouched

6. **Given** TheDev says "no worktree" (or similar) in their request
   **When** Phase 1b executes
   **Then** the old behavior is used: `git checkout -b QS_<N>` in the main directory
   **And** no worktree is created

2. **Given** a worktree exists for a story branch
   **When** TheDev runs quality gates in that worktree
   **Then** tests, ruff, mypy all execute against that worktree's code independently
   **And** results don't interfere with other worktrees

3. **Given** TheDev wants to run HA from a worktree
   **When** they launch HA in the worktree
   **Then** it uses the same config/ and other custom_components as the main worktree via symlinks
   **And** only quiet_solar code differs (it's the git-tracked code in this worktree's branch)

4. **Given** a story in a worktree is merged
   **When** TheDev cleans up
   **Then** the worktree is removed (symlinks go with it, originals untouched)
   **And** the main worktree's main branch is updated

5. **Given** the parallel workflow is documented
   **When** an agent (Claude or Cursor) reads development-lifecycle.md
   **Then** it has clear instructions for creating, using, and cleaning up worktrees
   **And** it knows what to symlink and why

## Tasks / Subtasks

- [x] Task 1: Document parallel development workflow in development-lifecycle.md (AC: #1, #2, #3, #4, #5)
  - [x] 1.1 Add new "Parallel Development with Worktrees" section
  - [x] 1.2 Document worktree creation with branch naming convention
  - [x] 1.3 Document symlink strategy: venv, config/, non-git custom_components
  - [x] 1.4 Document quality gate execution per worktree
  - [x] 1.5 Document cleanup: worktree removal after merge
  - [x] 1.6 Document directory convention for worktree placement
  - [x] 1.7 Document "no worktree" opt-out: user says "no worktree" → agent falls back to git checkout -b in main dir
  - [x] 1.8 Document caveats: no simultaneous HA instances on same config, shared venv means shared deps
- [x] Task 2: Create helper script for worktree lifecycle (AC: #1, #3, #4)
  - [x] 2.1 Create `scripts/worktree-setup.sh` -- creates worktree, branch, symlinks venv + config + non-git custom_components
  - [x] 2.2 Create `scripts/worktree-cleanup.sh` -- removes worktree after merge
- [x] Task 3: Update project-rules.md with parallel development reference (AC: #5)
  - [x] 3.1 Add worktree workflow link to project-rules.md

## Dev Notes

### Git Worktree Basics

Git worktrees allow multiple working directories linked to a single repo. Each worktree has its own checked-out branch and working tree, but shares the same `.git` object store. Only git-tracked files appear in a new worktree.

### Directory Convention

Worktrees live **outside** the main repo to avoid confusion:

```
~/Developer/homeassistant/
  quiet-solar/                          # main worktree (main branch)
    venv/                               # real venv (~GB, installed once)
    config/                             # real HA config (runtime state, not in git)
    custom_components/
      quiet_solar/                      # IN GIT - the project code
      hacs/                             # NOT in git - installed separately
      netatmo/                          # NOT in git
      ocpp/                             # NOT in git
      ...
  quiet-solar-worktrees/
    QS_42/                              # worktree for story on issue #42
      venv -> ../../quiet-solar/venv    # SYMLINK to main venv
      config -> ../../quiet-solar/config  # SYMLINK to main config
      custom_components/
        quiet_solar/                    # IN GIT - this branch's code (different!)
        hacs -> ../../../quiet-solar/custom_components/hacs  # SYMLINK
        netatmo -> ../../../quiet-solar/custom_components/netatmo  # SYMLINK
        ...
```

### Symlink Strategy: Why and What

**Problem**: A new git worktree only contains git-tracked files. For quiet-solar:
- `venv/` is not in git (~GB of HA + deps, expensive to recreate)
- `config/` is not in git (HA runtime state: database, storage, logs)
- `custom_components/hacs/`, `custom_components/netatmo/`, etc. are not in git (other HACS integrations)

**Solution**: Symlink these from the main worktree:

| Item | Why symlink | Needed for tests? | Needed for HA run? |
|------|-------------|-------------------|---------------------|
| `venv/` | Avoid ~GB duplicate install | YES | YES |
| `config/` | Share HA runtime config | NO (FakeHass) | YES |
| Non-git `custom_components/*` | Other integrations HA loads | NO | YES |

**Venv sharing is safe** because:
- All branches use the same `requirements.txt` / `requirements_test.txt`
- Tests import from `custom_components/quiet_solar/` directly (no editable pip install)
- If a branch changes requirements (rare), create a real venv for that worktree instead

### Caveats

1. **No simultaneous HA runs**: Two HA instances can't share the same `config/` (database locks). Run HA from one worktree at a time.
2. **Shared venv = shared deps**: If you `pip install` something in one worktree, all worktrees see it. This is fine for normal dev; only matters if experimenting with different dep versions.
3. **Symlink paths**: Use relative symlinks so the setup is portable if the parent directory moves.

### Worktree Setup Script Logic

```bash
#!/bin/bash
# scripts/worktree-setup.sh <issue_number>
# Creates worktree, branch, and symlinks

ISSUE=$1
MAIN_DIR=$(git rev-parse --show-toplevel)
WORKTREE_DIR="${MAIN_DIR}/../quiet-solar-worktrees/QS_${ISSUE}"

# Create worktree with new branch
git worktree add "$WORKTREE_DIR" -b "QS_${ISSUE}"

# Symlink venv
ln -s "${MAIN_DIR}/venv" "${WORKTREE_DIR}/venv"

# Symlink config
ln -s "${MAIN_DIR}/config" "${WORKTREE_DIR}/config"

# Symlink non-git custom_components
for dir in "${MAIN_DIR}/custom_components"/*/; do
  name=$(basename "$dir")
  if [ "$name" != "quiet_solar" ] && [ ! -e "${WORKTREE_DIR}/custom_components/${name}" ]; then
    ln -s "$dir" "${WORKTREE_DIR}/custom_components/${name}"
  fi
done
```

### Worktree Cleanup Script Logic

```bash
#!/bin/bash
# scripts/worktree-cleanup.sh <issue_number>
# Removes worktree after story is merged

ISSUE=$1
MAIN_DIR=$(git rev-parse --show-toplevel)
WORKTREE_DIR="${MAIN_DIR}/../quiet-solar-worktrees/QS_${ISSUE}"

git worktree remove "$WORKTREE_DIR"
# Symlinks are inside the worktree dir, so they're removed automatically
# The targets (main venv, config, etc.) are NOT affected
```

### Integration with Existing Workflows

Both `/bmad-dev-story` and `/bmad-quick-dev-new-preview` follow `development-lifecycle.md`. Phase 1b currently does `git checkout -b QS_<N>`. This story replaces Phase 1b with:

- **Default (worktree)**: `scripts/worktree-setup.sh <N>` → agent works in `../quiet-solar-worktrees/QS_<N>/`
- **Opt-out ("no worktree")**: `git checkout -b QS_<N>` → agent works in main dir (current behavior)

The agent detects "no worktree" (or similar phrasing) in the user's request and routes accordingly. The `project-rules.md` workflow routing table should mention this opt-out.

### What NOT to do

- Do NOT modify any production Python code in `custom_components/quiet_solar/`
- Do NOT modify test code
- Do NOT modify CI workflows -- worktrees are a local development concern
- Do NOT make worktrees mandatory -- they're an option for parallel work
- Do NOT create separate venvs per worktree by default (too expensive)
- Do NOT use absolute symlinks -- use relative paths for portability

### Previous story intelligence

**Story 1.1:** Established `_qsprocess/` as the canonical location for project process docs shared between Claude and Cursor.
**Story 1.4:** PR templates and issue templates exist. CODEOWNERS maps to @tmenguy.
**Recent:** development-lifecycle.md was updated with Phase 3d (code review) and Phase 3e (merge).

### Project Structure Notes

```
scripts/
  worktree-setup.sh     (new)
  worktree-cleanup.sh   (new)

_qsprocess/workflows/
  development-lifecycle.md   (modified -- new parallel development section)

_qsprocess/rules/
  project-rules.md           (modified -- add worktree reference)
```

### References

- [Source: _qsprocess/workflows/development-lifecycle.md] -- current lifecycle phases
- [Source: _qsprocess/rules/project-rules.md] -- project rules and documentation links
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 1] -- Automated Development Pipeline
- [Source: _bmad-output/planning-artifacts/prd.md#NFR23] -- minimal manual steps for developer workflows

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Updated `development-lifecycle.md` Phase 1b: worktree by default, "no worktree" opt-out
- Added Phase 3f: worktree cleanup after merge
- Added Appendix: Worktree Reference (directory layout, symlink table, caveats)
- Created `scripts/worktree-setup.sh` — creates worktree + symlinks (venv, config, non-git custom_components)
- Created `scripts/worktree-cleanup.sh` — removes worktree with --force (symlinks are untracked)
- Filters out `quiet_solar` (git-tracked) and `__pycache__` from symlink loop
- Updated `project-rules.md` — worktree mode note in routing, lifecycle section, merge quick ref
- End-to-end validated: setup script creates worktree, tests pass at 100% from worktree, cleanup removes it cleanly
- 3843 tests pass at 100% coverage, ruff clean, mypy clean

### Change Log

- 2026-03-20: Story 1.7 implemented — worktree workflow, helper scripts, docs updated

### File List

New files:
- `scripts/worktree-setup.sh`
- `scripts/worktree-cleanup.sh`

Modified files:
- `_qsprocess/workflows/development-lifecycle.md` (Phase 1b worktree default, Phase 3f cleanup, Appendix)
- `_qsprocess/rules/project-rules.md` (worktree mode note, lifecycle summary, merge quick ref)
- `_bmad-output/implementation-artifacts/1-7-parallel-story-development-worktrees.md` (story status + tasks)
