"""Dashboard generation and programmatic registration for Quiet Solar."""

import logging
import os
from typing import TYPE_CHECKING, Any

import yaml
import aiofiles
import aiofiles.os

from awesomeversion import AwesomeVersion
from homeassistant.components import frontend
from homeassistant.components.lovelace import dashboard as lovelace_dashboard
from homeassistant.components.lovelace.const import (
    CONF_ICON,
    CONF_REQUIRE_ADMIN,
    CONF_SHOW_IN_SIDEBAR,
    CONF_TITLE,
    CONF_URL_PATH,
    MODE_STORAGE,
)

# LOVELACE_DATA was introduced as a HassKey in HA 2025.2+; fall back to string
try:
    from homeassistant.components.lovelace.const import LOVELACE_DATA
except ImportError:
    LOVELACE_DATA = "lovelace"  # type: ignore[assignment]
from homeassistant.components.lovelace.resources import ResourceStorageCollection
from homeassistant.const import CONF_MODE, __version__ as HAVERSION
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
    store: Store[dict[str, Any]] = Store(
        hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY
    )
    return await store.async_load()


async def _async_save_dashboard_tracking(hass: HomeAssistant) -> None:
    """Persist which dashboards we created so we can restore them on restart."""
    store: Store[dict[str, Any]] = Store(
        hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY
    )
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
            lovelace_data.dashboards[url_path] = (
                lovelace_dashboard.LovelaceStorage(hass, config)
            )

            try:
                _register_panel(hass, dashboard_def)
                _LOGGER.info("Restored dashboard panel: %s", url_path)
            except ValueError:
                _LOGGER.warning(
                    "Failed to register panel for dashboard %s", url_path
                )

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

    tracking_store: Store[dict[str, Any]] = Store(
        hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY
    )
    await tracking_store.async_remove()


# ---------------------------------------------------------------------------
#  Auto-generate on first install
# ---------------------------------------------------------------------------


async def async_auto_generate_if_first_install(home: "QSHome") -> None:
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

    _LOGGER.info(
        "First install detected – auto-generating Quiet Solar dashboards"
    )
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
    """Generate a cache-busting tag from the current epoch."""
    import time
    return str(int(time.time()))


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


async def _async_update_resource(
    resources: ResourceStorageCollection,
    raw_name: str,
    url: str,
) -> None:
    """Create or update a single lovelace resource entry."""
    if not resources.loaded:
        await resources.async_load()

    for entry in resources.async_items():
        if (entry_url := entry["url"]).startswith(raw_name):
            if entry_url != url:
                await resources.async_update_item(entry["id"], {"url": url})
            return

    await resources.async_create_item({"res_type": "module", "url": url})


async def _async_copy_and_register_resources(
    from_dir: str,
    to_dir: str,
    namespace: str,
    tag: str,
    handler: ResourceStorageCollection | None,
) -> None:
    """Recursively copy resource files and register them with lovelace."""
    await aiofiles.os.makedirs(to_dir, exist_ok=True)
    for entry in await aiofiles.os.listdir(from_dir):
        src = os.path.join(from_dir, entry)
        dst = os.path.join(to_dir, entry)
        if await aiofiles.os.path.isfile(src):
            async with (
                aiofiles.open(src, "rb") as f_in,
                aiofiles.open(dst, "wb") as f_out,
            ):
                await f_out.write(await f_in.read())
            if handler is not None:
                await _async_update_resource(
                    handler,
                    f"{namespace}/{entry}",
                    f"{namespace}/{entry}?qs_tag={tag}",
                )
        elif await aiofiles.os.path.isdir(src):
            await _async_copy_and_register_resources(
                src, dst,
                f"{namespace}/{entry}", tag, handler,
            )


async def async_update_resources(hass: HomeAssistant) -> None:
    """Copy JS card resources to ``www/`` and register them with lovelace.

    This runs on *every* startup so that updated JS files from a component
    update are picked up immediately.  It does NOT touch dashboard content.
    """
    resources_dst = os.path.join(hass.config.path(), "www", DOMAIN)
    await aiofiles.os.makedirs(resources_dst, exist_ok=True)

    handler = _get_resource_handler_from_hass(hass)
    tag = _generate_qs_tag()
    namespace = _resource_namespace()

    try:
        await _async_copy_and_register_resources(
            _RESOURCES_DIR, resources_dst, namespace, tag, handler,
        )
        _LOGGER.debug("Dashboard JS resources updated")
    except FileNotFoundError:
        _LOGGER.debug("No bundled resources directory found, skipping")
    except Exception:
        _LOGGER.warning("Failed to update dashboard resources", exc_info=True)


# ---------------------------------------------------------------------------
#  Legacy wrappers – keep old function signatures working for any external
#  callers or tests that still use them.
# ---------------------------------------------------------------------------


def generate_dashboard_resource_qs_tag(home: "QSHome") -> str:
    """Generate a cache-busting tag (legacy wrapper)."""
    return _generate_qs_tag()


def generate_dashboard_resource_namespace(home: "QSHome") -> str:
    """Return the URL namespace (legacy wrapper)."""
    return _resource_namespace()


def _get_resource_handler(home: "QSHome") -> ResourceStorageCollection | None:
    """Return the lovelace ResourceStorageCollection (legacy wrapper)."""
    return _get_resource_handler_from_hass(home.hass)


async def update_resource(
    home: "QSHome",
    resources: ResourceStorageCollection,
    raw_name: str,
    url: str,
) -> None:
    """Create or update a single lovelace resource entry (legacy wrapper)."""
    await _async_update_resource(resources, raw_name, url)


# ---------------------------------------------------------------------------
#  Main entry point – called from the "Generate Dashboard" button
# ---------------------------------------------------------------------------


async def generate_dashboard_yaml(home: "QSHome") -> None:
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
        template_content: str = await hass.async_add_executor_job(
            _read_text_file, template_path
        )

        # Render with the HA template engine (gives access to all HA helpers)
        tpl = Template(template_content, hass)
        try:
            rendered = tpl.async_render(variables={"home": home})
        except TemplateError as err:
            _LOGGER.error(
                "Template error in %s: %s",
                dashboard_def["template_filename"], err,
                exc_info=True,
            )
            raise

        # Parse rendered YAML → Python dict
        try:
            lovelace_config = yaml.safe_load(rendered)
        except yaml.YAMLError as err:
            _LOGGER.error(
                "YAML parse error for %s: %s",
                dashboard_def["template_filename"], err,
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
        await _async_register_or_update_dashboard(
            hass, dashboard_def, lovelace_config
        )

    # Remember that dashboards exist so we can restore them on next restart
    await _async_save_dashboard_tracking(hass)

    # Also refresh JS resources
    await async_update_resources(hass)


def _read_text_file(path: str) -> str:
    """Read a text file synchronously (to be called via executor)."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()
