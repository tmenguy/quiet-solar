# Story 3.1: Failure Mode Catalog & Resilience Plan

Status: review

## Story

As TheDev,
I want a structured failure mode catalog documenting every external dependency weakness with its failure signature, planned system response, recovery path, and test coverage status,
So that resilience implementation is systematic and no dependency weakness is left unplanned.

## Acceptance Criteria

1. **Given** the failure mode catalog is created
   **When** TheDev consults it
   **Then** every external dependency is listed (solar forecast API, charger protocols, EV vendor APIs, HA state, numpy persistence, grid power, prediction confidence)
   **And** each entry has: failure signature, fallback behavior, recovery path, implementation status, test coverage status

2. **Given** the catalog is created
   **When** TheDev reviews the entries
   **Then** the catalog aligns with the architecture's resilience fallback table (AR5)
   **And** entries are prioritized by user impact (missed commitment > degraded optimization > reduced visibility)

3. **Given** the catalog is a living document
   **When** a new failure mode is discovered (NFR20)
   **Then** the catalog format supports easy updates with clear instructions for adding entries

## Tasks / Subtasks

- [x] Task 1: Create failure mode catalog document (AC: #1, #2)
  - [x] 1.1 Create `docs/failure-mode-catalog.md` with structured format
  - [x] 1.2 Document all 7 failure modes from AR5 with implementation status from codebase analysis
  - [x] 1.3 Add priority ranking by user impact
- [x] Task 2: Document existing implementation gaps (AC: #1)
  - [x] 2.1 Audit current codebase for existing failure handling and mark status per entry
  - [x] 2.2 Add gap analysis section identifying what Story 3.2 must implement
- [x] Task 3: Add catalog maintenance guidelines (AC: #3)
  - [x] 3.1 Add instructions for adding new failure modes and updating existing entries
  - [x] 3.2 Link catalog from project-rules.md or CLAUDE.md for discoverability

## Dev Notes

### Catalog Structure

Each failure mode entry must include these fields (from AC #1):

| Field | Description |
|---|---|
| **ID** | FM-001, FM-002, etc. |
| **Dependency** | External system name |
| **Priority** | P1 (missed commitment), P2 (degraded optimization), P3 (reduced visibility) |
| **Failure Signature** | How to detect the failure |
| **Fallback Behavior** | What the system does when failure is detected |
| **Recovery Path** | How normal operation resumes |
| **Implementation Status** | Implemented / Partial / Not Implemented |
| **Test Coverage** | Covered / Partial / None |

### AR5 Resilience Fallback Table (from architecture.md lines 567-575)

The architecture defines these 7 failure modes:

1. **Solar forecast API** (Solcast/OpenMeteo) — timeout/HTTP error → use last forecast → historical patterns if stale >6h → auto-retry next cycle
2. **Charger communication** — command not ACKed within 14s → retry 3x → mark unavailable → notify admin → auto-recover on response
3. **HA state unavailability** — entity = "unavailable"/"unknown" → filter invalid values → last-known-good → auto-recover on valid state
4. **Solver infeasibility** — no valid plan → safe defaults (mandatory priority, filler dropped, battery held) → re-solve next cycle
5. **Grid outage** — grid power = 0 → off-grid mode → emergency broadcast → auto-detect restoration
6. **Numpy persistence corruption** — corrupted .npy file → np.load defaults None → cold start → ring buffer rebuilds over time
7. **Prediction confidence degradation** — sparse history/pattern change → conservative defaults (higher SOC) → self-corrects over 30 days

### Codebase Analysis — Current Implementation Status

From scanning the actual code, here is what already exists:

| Failure Mode | Status | Evidence |
|---|---|---|
| Solar forecast API | **Partial** | `solar.py` has validation probe that filters invalid orchestrators on init, but no continuous monitoring or stale detection |
| Charger communication | **Partial** | OCPP notification handling with exception logging, wallbox fallback enum, but no retry loop or unavailable marking |
| HA state unavailability | **Implemented** | `bistate_duration.py`, `person.py`, `home.py` check STATE_UNKNOWN/STATE_UNAVAILABLE, filter invalid values |
| Solver infeasibility | **Partial** | Solver has constraint priority logic, but no explicit safe-default fallback on infeasibility |
| Grid outage | **Partial** | Off-grid mode code exists in `home.py`, but emergency broadcast not verified |
| Numpy persistence corruption | **Not Implemented** | No try/except around np.load calls observed |
| Prediction confidence | **Partial** | Conservative defaults exist in person.py, but no explicit confidence scoring |

### Existing Failure Handling Patterns in Codebase

The code uses these patterns (Story 3.2 should follow the same style):

1. **Broad try/except with logging**: `except Exception as e: _LOGGER.error("...: %s", e, exc_info=True, stack_info=True)`
2. **Import fallbacks**: try/except ImportError with fallback class (wallbox enum)
3. **State validation**: check for STATE_UNKNOWN/STATE_UNAVAILABLE before processing
4. **Validation probes**: test external APIs at init, filter invalid ones
5. **Re-entry lock detection**: skip concurrent updates with asyncio.Lock

### Notable Gaps (Feed into Story 3.2)

- No HTTP timeout configuration for external API calls
- No retry logic with exponential backoff (NFR12 requires this)
- No circuit breaker pattern for continuous dependency monitoring
- No stale data detection for solar forecasts (AR5 specifies >6h threshold)
- No explicit infeasibility handler in solver
- Bare `except:` clauses in constraints.py (should catch specific exceptions)

### Output Location

Create catalog at `docs/failure-mode-catalog.md`. This aligns with the `project_knowledge` config path (`{project-root}/docs`).

### What NOT to do

- Do NOT implement any resilience code — that's Story 3.2
- Do NOT write tests — that's Story 3.3
- Do NOT modify any production Python code
- Do NOT modify existing architecture documents
- Keep the catalog concise and actionable — it's a reference for dev agents, not a novel

### Previous story intelligence

**Story 1.2:** CI pipeline established with 4 parallel quality gate jobs. Timezone must be Europe/Paris.
**Story 1.4:** PR templates and issue templates now exist. CODEOWNERS maps to @tmenguy.

### Project Structure Notes

```
docs/
└── failure-mode-catalog.md    (new)
```

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Decision 1] — AR5 resilience fallback table (lines 563-575)
- [Source: _bmad-output/planning-artifacts/architecture.md#Gap Analysis] — Gap #3: Failure Mode / Resilience Architecture (lines 359-365)
- [Source: _bmad-output/planning-artifacts/architecture.md#External Integration Points] — Integration table (lines 965-975)
- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.1] — acceptance criteria (lines 409-424)
- [Source: _bmad-output/planning-artifacts/prd.md#FR19-FR25b] — Resilience & Failure Handling (lines 461-480)
- [Source: _bmad-output/planning-artifacts/prd.md#NFR12] — Retry with exponential backoff
- [Source: _bmad-output/planning-artifacts/prd.md#NFR13] — Failure detection within evaluation cycle
- [Source: _bmad-output/planning-artifacts/prd.md#NFR20] — Catalog must be updated on new failure modes

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Created failure mode catalog with 7 entries (FM-001 through FM-007) aligned with AR5
- Each entry has: failure signature, fallback behavior, recovery path, implementation status, test coverage
- Priority ranking: 3x P1 (charger, solver, grid), 2x P2 (solar forecast, prediction), 2x P3 (HA state, numpy)
- Codebase audit corrected initial assumptions: numpy persistence IS handled (bare except), off-grid IS fully implemented
- Gap analysis identifies 8 items for Story 3.2 implementation
- Entry template and maintenance instructions included for NFR20 compliance
- Linked from project-rules.md for discoverability
- 3843 tests pass at 100% coverage, no regressions

### Change Log

- 2026-03-20: Story 3.1 implemented — failure mode catalog created

### File List

New files:
- `docs/failure-mode-catalog.md`

Modified files:
- `_qsprocess/rules/project-rules.md` (added catalog link to Full Documentation section)
- `_bmad-output/implementation-artifacts/3-1-failure-mode-catalog-resilience-plan.md` (story status + tasks)
