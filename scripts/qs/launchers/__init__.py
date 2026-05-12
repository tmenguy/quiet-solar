"""Harness-specific launcher emitters.

Each launcher module exposes ``build_payload(work_dir, issue, title, *,
next_cmd, next_prompt=None)`` returning a dict with at minimum the keys
``tool``, ``same_context``, and ``new_context``.

The dispatcher in :mod:`scripts.qs.next_step` selects the correct
launcher based on :func:`scripts.qs.harness.detect`.
"""
