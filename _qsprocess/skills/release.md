# /release

Create a release: determine tag, bump version, commit, tag, push.

## Prerequisites

Must be on `main` branch, up to date.

## Steps

### 1. Create release

```bash
python scripts/qs/release.py
```

### 2. Report

Show the tag and version. Remind the user that GitHub Actions will:
- Run the full test suite
- Validate HACS compatibility
- Create the GitHub Release with changelog

Direct them to the Actions tab to monitor.
