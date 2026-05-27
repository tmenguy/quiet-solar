"""Dashboard generation and programmatic registration for Quiet Solar."""

import asyncio
import contextlib
import itertools
import logging
import os
import re
from typing import TYPE_CHECKING, Any

import aiofiles
import aiofiles.os
import yaml
from awesomeversion import AwesomeVersion
from homeassistant.components import frontend
from homeassistant.components.lovelace import dashboard as lovelace_dashboard

# LOVELACE_DATA was introduced as a HassKey in HA 2025.2+; fall back to string
from homeassistant.components.lovelace.const import (
    CONF_ICON,
    CONF_REQUIRE_ADMIN,
    CONF_SHOW_IN_SIDEBAR,
    CONF_TITLE,
    CONF_URL_PATH,
    LOVELACE_DATA,
    MODE_STORAGE,
)
from homeassistant.components.lovelace.resources import ResourceStorageCollection
from homeassistant.const import CONF_MODE
from homeassistant.const import __version__ as HAVERSION
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.storage import Store
from homeassistant.helpers.template import Template

from ..const import DOMAIN

if TYPE_CHECKING:
    from ..ha_model.home import QSHome

_LOGGER = logging.getLogger(__name__)

# Storage version and key for tracking our dashboard registrations
QS_DASHBOARDS_STORAGE_VERSION = 1
QS_DASHBOARDS_STORAGE_KEY = f"{DOMAIN}_dashboards"

# Path to bundled JS resources, relative to this file
_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")

# Dashboard definitions: Jinja2 template → storage-mode Lovelace dashboard
DASHBOARD_CUSTOM = {
    "id": "quiet-solar-qs",
    "url_path": "quiet-solar",
    "title": "Quiet Solar",
    "icon": "mdi:solar-power",
    "show_in_sidebar": True,
    "require_admin": False,
    "template_filename": "quiet_solar_dashboard_template.yaml.j2",
}

DASHBOARD_STANDARD = {
    "id": "quiet-solar-standard-qs",
    "url_path": "quiet-solar-standard",
    "title": "Quiet Std",
    "icon": "mdi:solar-power",
    "show_in_sidebar": True,
    "require_admin": False,
    "template_filename": "quiet_solar_dashboard_template_standard_ha.yaml.j2",
}

ALL_DASHBOARDS = [DASHBOARD_CUSTOM, DASHBOARD_STANDARD]


# ---------------------------------------------------------------------------
#  Lovelace panel / storage helpers
# ---------------------------------------------------------------------------


def _get_lovelace_data(hass: HomeAssistant):
    """Return the LovelaceData object, or None if lovelace is not ready."""
    return hass.data.get(LOVELACE_DATA)


def _register_panel(
    hass: HomeAssistant,
    dashboard_def: dict[str, Any],
    *,
    update: bool = False,
) -> None:
    """Register a lovelace panel for the given dashboard definition."""
    kwargs: dict[str, Any] = {
        "frontend_url_path": dashboard_def["url_path"],
        "require_admin": dashboard_def.get("require_admin", False),
        "config": {"mode": MODE_STORAGE},
        "update": update,
    }
    if dashboard_def.get("show_in_sidebar", True):
        kwargs["sidebar_title"] = dashboard_def["title"]
        kwargs["sidebar_icon"] = dashboard_def.get("icon", "mdi:view-dashboard")

    frontend.async_register_built_in_panel(hass, "lovelace", **kwargs)


def _make_lovelace_config(dashboard_def: dict[str, Any]) -> dict[str, Any]:
    """Build a config dict that LovelaceStorage expects."""
    return {
        "id": dashboard_def["id"],
        CONF_URL_PATH: dashboard_def["url_path"],
        CONF_TITLE: dashboard_def["title"],
        CONF_ICON: dashboard_def.get("icon", "mdi:view-dashboard"),
        CONF_SHOW_IN_SIDEBAR: dashboard_def.get("show_in_sidebar", True),
        CONF_REQUIRE_ADMIN: dashboard_def.get("require_admin", False),
        CONF_MODE: MODE_STORAGE,
    }


# ---------------------------------------------------------------------------
#  Tracking storage helpers
# ---------------------------------------------------------------------------


async def _async_load_tracking(hass: HomeAssistant) -> dict[str, Any] | None:
    """Load our tracking storage (returns None on first install)."""
    store: Store[dict[str, Any]] = Store(hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY)
    return await store.async_load()


async def _async_save_dashboard_tracking(hass: HomeAssistant) -> None:
    """Persist which dashboards we created so we can restore them on restart."""
    store: Store[dict[str, Any]] = Store(hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY)
    await store.async_save({"dashboards": [d["id"] for d in ALL_DASHBOARDS]})


def _dashboards_ever_generated(tracking_data: dict[str, Any] | None) -> bool:
    """Return True if dashboards were generated at least once before."""
    return tracking_data is not None and bool(tracking_data.get("dashboards"))


# ---------------------------------------------------------------------------
#  Startup / teardown
# ---------------------------------------------------------------------------


async def async_restore_dashboards_and_update_resources(
    hass: HomeAssistant,
) -> None:
    """Called once from ``async_setup`` on every HA start.

    1. Re-register dashboard panels that were generated in a previous session
       so they appear in the sidebar immediately.
    2. Always refresh the JS card resources (they may have been updated with a
       new version of the component), but do NOT touch the dashboard *content*
       to preserve any manual edits the user may have made.
    """
    lovelace_data = _get_lovelace_data(hass)
    if lovelace_data is None:
        _LOGGER.debug("Lovelace not ready, skipping dashboard restore")
        return

    tracking_data = await _async_load_tracking(hass)

    # --- Restore panels from tracking data ---
    if _dashboards_ever_generated(tracking_data):
        registered_ids: list[str] = tracking_data["dashboards"]  # type: ignore[index]

        for dashboard_def in ALL_DASHBOARDS:
            if dashboard_def["id"] not in registered_ids:
                continue

            url_path = dashboard_def["url_path"]

            if url_path in lovelace_data.dashboards:
                continue  # already registered

            panels = hass.data.get(frontend.DATA_PANELS, {})
            if url_path in panels:
                _LOGGER.warning(
                    "Panel %s already exists, cannot register dashboard",
                    url_path,
                )
                continue

            config = _make_lovelace_config(dashboard_def)
            lovelace_data.dashboards[url_path] = lovelace_dashboard.LovelaceStorage(hass, config)

            try:
                _register_panel(hass, dashboard_def)
                _LOGGER.info("Restored dashboard panel: %s", url_path)
            except ValueError:
                _LOGGER.warning("Failed to register panel for dashboard %s", url_path)

    # --- Always refresh JS card resources ---
    await async_update_resources(hass)


async def async_unregister_dashboards(hass: HomeAssistant) -> None:
    """Remove all Quiet Solar dashboard panels and their stored content."""
    lovelace_data = _get_lovelace_data(hass)

    for dashboard_def in ALL_DASHBOARDS:
        url_path = dashboard_def["url_path"]
        frontend.async_remove_panel(hass, url_path, warn_if_unknown=False)

        if lovelace_data is not None and url_path in lovelace_data.dashboards:
            lovelace_store = lovelace_data.dashboards.pop(url_path)
            await lovelace_store.async_delete()

    tracking_store: Store[dict[str, Any]] = Store(hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY)
    await tracking_store.async_remove()


# ---------------------------------------------------------------------------
#  Auto-generate on first install
# ---------------------------------------------------------------------------


async def async_auto_generate_if_first_install(home: QSHome) -> None:
    """Generate dashboards automatically when the integration is installed
    for the first time (i.e. no tracking data exists yet).

    Called from the data handler after the home device is fully created.
    On subsequent startups this is a no-op because tracking data exists.
    """
    hass = home.hass
    tracking_data = await _async_load_tracking(hass)

    if _dashboards_ever_generated(tracking_data):
        _LOGGER.debug("Dashboards already generated, skipping auto-generation")
        return

    _LOGGER.info("First install detected – auto-generating Quiet Solar dashboards")
    await generate_dashboard_yaml(home)


# ---------------------------------------------------------------------------
#  Register / update a single dashboard
# ---------------------------------------------------------------------------


async def _async_register_or_update_dashboard(
    hass: HomeAssistant,
    dashboard_def: dict[str, Any],
    lovelace_config_dict: dict[str, Any],
) -> None:
    """Create the panel if needed and save the lovelace config content."""
    lovelace_data = _get_lovelace_data(hass)
    if lovelace_data is None:
        _LOGGER.error("Lovelace data not available, cannot register dashboard")
        return

    url_path = dashboard_def["url_path"]

    if url_path in lovelace_data.dashboards:
        lovelace_store = lovelace_data.dashboards[url_path]
    else:
        config = _make_lovelace_config(dashboard_def)
        lovelace_store = lovelace_dashboard.LovelaceStorage(hass, config)
        lovelace_data.dashboards[url_path] = lovelace_store

        try:
            _register_panel(hass, dashboard_def)
        except ValueError:
            # Panel URL might already exist (e.g. leftover from config.yaml)
            try:
                _register_panel(hass, dashboard_def, update=True)
            except ValueError:
                _LOGGER.error("Cannot register dashboard panel %s", url_path)
                return

    await lovelace_store.async_save(lovelace_config_dict)


# ---------------------------------------------------------------------------
#  Lovelace resource handling (JS custom-card files)
#  These functions accept ``hass`` so they can run at startup without a
#  ``home`` object.
# ---------------------------------------------------------------------------


def _generate_qs_tag() -> str:
    """Generate a cache-busting tag from a high-resolution clock.

    QS-199 review-fix S11 — `time.time_ns()` (nanoseconds) avoids the
    identical-tag collision that `int(time.time())` would produce on two
    restarts within the same wall-clock second (browsers would then serve
    cached content despite changed files). Still a digit string.
    """
    import time

    return str(time.time_ns())


# QS-199 A1b / review-fix M6 + S8 + S9 + #02 N1: matches a relative `.js`
# import URL in any of three forms:
#   - static:       `from <q>./...<q>`
#   - dynamic:      `import(<q>./...<q>)`
#   - bare side-fx: `import <q>./...<q>`  (no `from`, no `(`)  [#02 N1]
# The path alternation `\./[^'"?\s]+\.js` matches BOTH `./shared/x.js`
# (top-level cards) AND `./x.js` (sibling imports inside `shared/` — M6).
# `\bfrom\s*` / `\bimport\s*\(\s*` tolerate the no-whitespace minified
# form `from'./x.js'` (S9) and word-boundary-guard against `transform`.
# The bare-import alternative `\bimport\s+(?=['"])` requires `import`
# directly followed by a quote, so it can't swallow a named import
# (`import {X} from ...`, where `import ` is followed by `{`). The
# optional `(?P<query>\?...)` captures any pre-existing query string so
# the replacement can merge rather than clobber it (S10).
_IMPORT_REWRITE_RE = re.compile(
    r"""(?P<lead>\bfrom\s*|\bimport\s*\(\s*|\bimport\s+(?=['"]))"""
    r"""(?P<q>['"])"""
    r"""(?P<path>\./[^'"?\s]+\.js)"""
    r"""(?P<query>\?[^'"]*)?"""
    r"""(?P=q)"""
)


def _merge_qs_tag(existing_query: str, tag: str) -> str:
    """Merge `qs_tag=<tag>` into an existing query string.

    QS-199 review-fix S10 — preserves any pre-existing non-`qs_tag` params
    (e.g. `?v=1&theme=dark`), replacing only the `qs_tag` value (or
    appending it when absent). Always overwrites a stale `qs_tag` (R1
    idempotency).

    QS-199 review-fix #02 N4 — a valueless param (`?flag`) is normalised
    to `flag=` by `urlencode`. This is cosmetic and never occurs in the
    dev-authored import URLs (they carry no query string at all), so it
    is left as-is.
    """
    from urllib.parse import parse_qsl, urlencode

    params = [(k, v) for (k, v) in parse_qsl(existing_query.lstrip("?"), keep_blank_values=True) if k != "qs_tag"]
    params.append(("qs_tag", tag))
    return "?" + urlencode(params)


def _rewrite_shared_imports(content: str, tag: str) -> str:
    """Inject `?qs_tag=<tag>` into every relative `.js` import URL.

    QS-199 review-fix S4 — uses a callable replacement so `tag` is treated
    as a literal string (a tag containing a backslash can't be mis-parsed
    as a regex backreference). Always overwrites a pre-existing `qs_tag`;
    preserves other query params (S10).
    """

    def _repl(m: re.Match[str]) -> str:
        new_query = _merge_qs_tag(m.group("query") or "", tag)
        return f"{m.group('lead')}{m.group('q')}{m.group('path')}{new_query}{m.group('q')}"

    return _IMPORT_REWRITE_RE.sub(_repl, content)


def _maybe_rewrite_js_bytes(raw: bytes, tag: str, name: str) -> bytes:
    """Decode + rewrite a `.js` file's import URLs, or byte-copy on failure.

    QS-199 review-fix M7 — a `.js` file with non-UTF-8 bytes (Latin-1,
    BOM mojibake) must NOT abort the whole copy. On `UnicodeDecodeError`
    we fall back to returning the raw bytes unchanged so the destination
    tree stays complete and every other resource still registers.
    """
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        _LOGGER.warning("Non-UTF-8 JS resource %s copied without import rewrite", name)
        return raw
    return _rewrite_shared_imports(content, tag).encode("utf-8")


def _resource_namespace() -> str:
    """Return the URL namespace for dashboard resources."""
    return f"/local/{DOMAIN}"


def _get_resource_handler_from_hass(
    hass: HomeAssistant,
) -> ResourceStorageCollection | None:
    """Return the lovelace ResourceStorageCollection, or None."""
    if not (hass_data := hass.data):
        return None

    if (lovelace_data := hass_data.get("lovelace")) is None:
        return None

    if AwesomeVersion(HAVERSION) > "2025.1.99":
        resources = lovelace_data.resources
    else:
        resources = lovelace_data.get("resources")

    if resources is None:
        return None

    if not hasattr(resources, "store") or resources.store is None:
        return None

    if resources.store.key != "lovelace_resources" or resources.store.version != 1:
        return None

    return resources


def _url_path_without_query(url: str) -> str:
    """Return the URL up to (but excluding) the `?` query string.

    QS-199 review-fix #02 N5 — `?` is the query delimiter for the
    `/local/quiet_solar/...` resource URLs we generate (they never embed
    a literal `?` in the path), so splitting on the first `?` cleanly
    separates path from the `?qs_tag=...` cache-buster.
    """
    return url.split("?", 1)[0]


# QS-199 review-fix #02 N2 — JS module extensions whose imports get the
# cache-buster rewrite. `.mjs` is included for forward-compat even though
# the bundled tree is `.js`-only today.
_JS_MODULE_EXTENSIONS = (".js", ".mjs")


async def _async_update_resource(
    resources: ResourceStorageCollection,
    raw_name: str,
    url: str,
) -> None:
    """Create or update a single lovelace resource entry.

    QS-199 review-fix N9 — match on the resource URL's path-up-to-`?`
    EXACTLY equal to `raw_name` instead of a bare `startswith` prefix.
    The old prefix match would also match a user's manual backup URL
    (e.g. `.../qs-car-card.js.backup`) and silently overwrite it.
    """
    if not resources.loaded:
        await resources.async_load()

    for entry in resources.async_items():
        if _url_path_without_query(entry["url"]) == raw_name:
            if entry["url"] != url:
                await resources.async_update_item(entry["id"], {"url": url})
            return

    await resources.async_create_item({"res_type": "module", "url": url})


# Monotonic per-process counter feeding the unique atomic-write temp name (S1).
_QSTMP_COUNTER = itertools.count()

# Suffix of the atomic-write temp files (S1) — also the orphan-sweep glob (N2).
_QSTMP_SUFFIX = ".qstmp"


async def _async_remove_orphan_temps(root: str) -> None:
    """Reclaim orphan ``*.qstmp`` temp files under ``root`` (recursively).

    QS-199 review-fix #03 N2 — with the unique per-write temp name (S1) a
    hard kill (SIGKILL / OOM / power loss) between the write and the
    ``os.replace`` leaves a uniquely-named orphan that the in-process
    ``OSError`` cleanup can't reach, so they accumulate across crashes.
    A one-time sweep at the top of ``async_update_resources`` reclaims
    them. Best-effort: individual removal failures are suppressed.
    """
    if not await aiofiles.os.path.isdir(root):
        return
    for entry in await aiofiles.os.listdir(root):
        path = os.path.join(root, entry)
        if await aiofiles.os.path.isdir(path):
            await _async_remove_orphan_temps(path)
        elif entry.endswith(_QSTMP_SUFFIX):
            with contextlib.suppress(OSError):
                await aiofiles.os.remove(path)


async def _async_atomic_write_bytes(dst: str, data: bytes) -> None:
    """Write `data` to `dst` atomically (temp file + os.replace).

    QS-199 review-fix S12 — writing in place risks a truncated file if the
    disk fills or HA is killed mid-write; the browser would then load a
    JS syntax error. Writing to a temp sibling and `os.replace`-ing it
    into place makes the swap atomic on the same filesystem.

    QS-199 review-fix #02 S1 — the temp name is unique per call
    (`<pid>.<counter>`). `async_update_resources` is awaited without a
    lock from both startup and the "Generate Dashboard" button; a shared
    `.qstmp` name would let two interleaved runs corrupt the temp or have
    one run's `os.replace` race against the other renaming the same temp
    away (surfacing as a spurious `FileNotFoundError`).

    QS-199 review-fix #02 N3 — on a mid-write failure (disk full, replace
    race) the partial temp is unlinked so orphans don't accumulate; `dst`
    keeps its old content (atomicity holds).
    """
    tmp = f"{dst}.{os.getpid()}.{next(_QSTMP_COUNTER)}{_QSTMP_SUFFIX}"
    try:
        async with aiofiles.open(tmp, "wb") as f_out:
            await f_out.write(data)
        await aiofiles.os.replace(tmp, dst)
    except OSError:
        with contextlib.suppress(OSError):
            await aiofiles.os.remove(tmp)
        raise


async def _async_copy_and_register_resources(
    from_dir: str,
    to_dir: str,
    namespace: str,
    tag: str,
    handler: ResourceStorageCollection | None,
    *,
    is_top_level: bool = True,
) -> None:
    """Recursively copy resource files and register them with lovelace.

    QS-199:
      * A1a — Only top-level files are registered as Lovelace resources;
        subdirectory files (e.g. `shared/*.js`) are still copied but skip
        registration. Shared modules are ES-module dependencies imported
        by the top-level cards, not entry-point cards themselves.
      * A1b / M6 — `.js` files have their relative import URLs (both
        `from './shared/<path>.js'` in top-level cards AND `./sibling.js`
        between files inside `shared/`) regex-rewritten to inject
        `?qs_tag=<tag>`, using the same `tag` value as the Lovelace
        registration URL. Always overwrites a pre-existing `qs_tag`
        (idempotent across HA restarts); preserves other query params.
      * M7 — a non-UTF-8 `.js` file is byte-copied (no rewrite) instead
        of aborting the recursion.
      * S12 — files are written to a `.qstmp` sibling then atomically
        `os.replace()`d into place so a disk-full / kill mid-write can't
        leave a truncated JS file that the browser would choke on.
      * N8 — subdirectories (e.g. `shared/`) are copied BEFORE the
        top-level files that import them, so a browser fetching a card
        mid-copy never resolves its inheritance chain through a
        not-yet-rewritten shared module.
    """
    await aiofiles.os.makedirs(to_dir, exist_ok=True)

    entries = await aiofiles.os.listdir(from_dir)
    files: list[str] = []
    subdirs: list[str] = []
    for entry in entries:
        if await aiofiles.os.path.isfile(os.path.join(from_dir, entry)):
            files.append(entry)
        elif await aiofiles.os.path.isdir(os.path.join(from_dir, entry)):
            subdirs.append(entry)

    # N8: dependency order — recurse into subdirectories (shared modules)
    # first, then copy the top-level files that import them.
    for entry in subdirs:
        await _async_copy_and_register_resources(
            os.path.join(from_dir, entry),
            os.path.join(to_dir, entry),
            f"{namespace}/{entry}",
            tag,
            handler,
            is_top_level=False,
        )

    for entry in files:
        src = os.path.join(from_dir, entry)
        dst = os.path.join(to_dir, entry)
        async with aiofiles.open(src, "rb") as f_in:
            raw = await f_in.read()
        # M7 / #02 N2: only JS-module files (.js / .mjs) are decoded +
        # rewritten; the helper falls back to the raw bytes on a
        # UnicodeDecodeError. Everything else is byte-copied verbatim.
        is_js_module = entry.endswith(_JS_MODULE_EXTENSIONS)
        out_bytes = _maybe_rewrite_js_bytes(raw, tag, entry) if is_js_module else raw
        await _async_atomic_write_bytes(dst, out_bytes)
        # A1a: skip lovelace registration for files in subdirectories
        if handler is not None and is_top_level:
            await _async_update_resource(
                handler,
                f"{namespace}/{entry}",
                f"{namespace}/{entry}?qs_tag={tag}",
            )
        elif handler is not None and not is_top_level:
            _LOGGER.debug("Skipped lovelace registration for shared module %s/%s", namespace, entry)


# QS-199 review-fix #04 ES2 — serialize the whole resource update. The two
# entry points (startup restore in `async_restore_dashboards_and_update_resources`
# and the "Generate Dashboard" button via `generate_dashboard_yaml`) are
# otherwise unlocked and can interleave their sweep + copy + register steps,
# which re-introduced the temp-file race classes (round-2 S1, round-3 N2/#04
# ES2). One module-level lock kills the entire class at the root; the unique
# temp name (S1) stays as defense-in-depth.
_RESOURCE_UPDATE_LOCK = asyncio.Lock()


async def async_update_resources(hass: HomeAssistant) -> None:
    """Copy JS card resources to ``www/`` and register them with lovelace.

    This runs on *every* startup so that updated JS files from a component
    update are picked up immediately.  It does NOT touch dashboard content.

    QS-199 review-fix #04 ES2 — the body runs under `_RESOURCE_UPDATE_LOCK`
    so a concurrent invocation can't sweep away (or os.replace over) a temp
    file the other invocation is mid-write on.
    """
    async with _RESOURCE_UPDATE_LOCK:
        await _async_update_resources_locked(hass)


async def _async_update_resources_locked(hass: HomeAssistant) -> None:
    """Body of `async_update_resources`, run while holding the update lock."""
    resources_dst = os.path.join(hass.config.path(), "www", DOMAIN)
    await aiofiles.os.makedirs(resources_dst, exist_ok=True)

    # QS-199 review-fix #03 N2 — reclaim any orphan *.qstmp temp files left
    # behind by a prior hard kill before copying fresh resources. Safe under
    # the update lock: no concurrent invocation can be mid-write (#04 ES2).
    await _async_remove_orphan_temps(resources_dst)

    # QS-199 review-fix #02 S1 — check the SOURCE tree up front instead of
    # catching `FileNotFoundError` around the whole copy. A per-file
    # `os.replace` race (now also mitigated by unique temp names) would
    # otherwise raise `FileNotFoundError` mid-recursion and be mis-logged
    # as "No bundled resources directory found", silently skipping the
    # remaining files.
    if not await aiofiles.os.path.isdir(_RESOURCES_DIR):
        _LOGGER.debug("No bundled resources directory found, skipping")
        return

    handler = _get_resource_handler_from_hass(hass)
    tag = _generate_qs_tag()
    namespace = _resource_namespace()

    try:
        await _async_copy_and_register_resources(
            _RESOURCES_DIR,
            resources_dst,
            namespace,
            tag,
            handler,
        )
        _LOGGER.debug("Dashboard JS resources updated")
    except Exception:
        _LOGGER.warning("Failed to update dashboard resources", exc_info=True)


# ---------------------------------------------------------------------------
#  Legacy wrappers – keep old function signatures working for any external
#  callers or tests that still use them.
# ---------------------------------------------------------------------------


def generate_dashboard_resource_qs_tag(home: QSHome) -> str:
    """Generate a cache-busting tag (legacy wrapper)."""
    return _generate_qs_tag()


def generate_dashboard_resource_namespace(home: QSHome) -> str:
    """Return the URL namespace (legacy wrapper)."""
    return _resource_namespace()


def _get_resource_handler(home: QSHome) -> ResourceStorageCollection | None:
    """Return the lovelace ResourceStorageCollection (legacy wrapper)."""
    return _get_resource_handler_from_hass(home.hass)


async def update_resource(
    home: QSHome,
    resources: ResourceStorageCollection,
    raw_name: str,
    url: str,
) -> None:
    """Create or update a single lovelace resource entry (legacy wrapper)."""
    await _async_update_resource(resources, raw_name, url)


# ---------------------------------------------------------------------------
#  Main entry point – called from the "Generate Dashboard" button
# ---------------------------------------------------------------------------


async def generate_dashboard_yaml(home: QSHome) -> None:
    """Render Jinja2 templates and push the result into Lovelace storage.

    The templates produce YAML which is parsed to a dict and saved directly
    into storage-mode Lovelace dashboards.  No YAML files on disk and no
    ``configuration.yaml`` entries are needed.
    """
    hass = home.hass
    base_dir = os.path.dirname(__file__)

    _LOGGER.info("Regenerating Quiet Solar dashboards")

    for dashboard_def in ALL_DASHBOARDS:
        template_path = os.path.join(base_dir, dashboard_def["template_filename"])

        # Read the Jinja2 template (blocking I/O → executor)
        template_content: str = await hass.async_add_executor_job(_read_text_file, template_path)

        # Render with the HA template engine (gives access to all HA helpers)
        tpl = Template(template_content, hass)
        try:
            rendered = tpl.async_render(variables={"home": home})
        except TemplateError as err:
            _LOGGER.error(
                "Template error in %s: %s",
                dashboard_def["template_filename"],
                err,
                exc_info=True,
            )
            raise

        # Parse rendered YAML → Python dict
        try:
            lovelace_config = yaml.safe_load(rendered)
        except yaml.YAMLError as err:
            _LOGGER.error(
                "YAML parse error for %s: %s",
                dashboard_def["template_filename"],
                err,
                exc_info=True,
            )
            raise

        if not isinstance(lovelace_config, dict):
            _LOGGER.error(
                "Template %s did not produce a valid dict",
                dashboard_def["template_filename"],
            )
            continue

        # Push into a storage-mode Lovelace dashboard
        await _async_register_or_update_dashboard(hass, dashboard_def, lovelace_config)

    # Remember that dashboards exist so we can restore them on next restart
    await _async_save_dashboard_tracking(hass)

    # Also refresh JS resources
    await async_update_resources(hass)


def _read_text_file(path: str) -> str:
    """Read a text file synchronously (to be called via executor)."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()
