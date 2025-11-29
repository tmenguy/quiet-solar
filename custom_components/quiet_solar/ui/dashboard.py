import logging
import os
from typing import TYPE_CHECKING

from homeassistant.components.lovelace.resources import ResourceStorageCollection


from awesomeversion import AwesomeVersion
from homeassistant.const import Platform, __version__ as HAVERSION

import aiofiles.os
import aiofiles
from homeassistant.helpers.template import Template
from homeassistant.exceptions import TemplateError

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def generate_dashboard_resource_qs_tag(home) -> str:
    """ generate a tag from now epoch seconds to force browser reload of resource"""
    import time
    return str(int(time.time()))


def generate_dashboard_resource_namespace(home) -> str:
    """Get the dashboard resource namespace."""
    return f"/local/{DOMAIN}"



def _get_resource_handler(home) -> ResourceStorageCollection | None:
    """Get the resource handler."""
    resources: ResourceStorageCollection | None
    if not (hass_data := home.hass.data):
        _LOGGER.error("_get_resource_handler: Can not access the hass data")
        return None

    if (lovelace_data := hass_data.get("lovelace")) is None:
        _LOGGER.warning("_get_resource_handler: Can not access the lovelace integration data", )
        return None

    if AwesomeVersion(HAVERSION) > "2025.1.99":
        # Changed to 2025.2.0
        # Changed in https://github.com/home-assistant/core/pull/136313
        resources = lovelace_data.resources
    else:
        resources = lovelace_data.get("resources")

    if resources is None:
        _LOGGER.warning("_get_resource_handler: Can not access the dashboard resources")
        return None

    if not hasattr(resources, "store") or resources.store is None:
        _LOGGER.info("_get_resource_handler: YAML mode detected, can not update resources")
        return None

    if resources.store.key != "lovelace_resources" or resources.store.version != 1:
        _LOGGER.warning("_get_resource_handler: Can not use the dashboard resources")
        return None

    return resources

async def update_resource(home, resources, raw_name, url):

    """Update dashboard resources."""

    if not resources.loaded:
        await resources.async_load()

    for entry in resources.async_items():
        if (entry_url := entry["url"]).startswith(raw_name):
            if entry_url != url:
                _LOGGER.info(
                    "update_resource: Updating existing dashboard resource from %s to %s",
                    entry_url,
                    url,
                )
                await resources.async_update_item(entry["id"], {"url": url})
            return

    # Nothing was updated, add the resource
    _LOGGER.info("update_resource: Adding dashboard resource %s", url)
    await resources.async_create_item({"res_type": "module", "url": url})



async def generate_dashboard_yaml(home):
    base_dir = os.path.join(os.path.dirname(__file__))
    jinja_path = os.path.join(base_dir, "quiet_solar_dashboard_template.yaml.j2")
    jinja_path_standard = os.path.join(base_dir, "quiet_solar_dashboard_template_standard_ha.yaml.j2")

    _LOGGER.warning("generate_dashboard_yaml: regenerating dashboard YAML file")

    # Async, non-blocking read of the Jinja template
    async with aiofiles.open(jinja_path, "r") as f:
        template_content = await f.read()

    async with aiofiles.open(jinja_path_standard, "r") as f:
        template_content_standard = await f.read()

    tpl = Template(template_content, home.hass)
    tpl_std = Template(template_content_standard, home.hass)

    try:
        rendered = tpl.async_render(variables={"home": home})
        rendered_std = tpl_std.async_render(variables={"home": home})

        storage_path: str = os.path.join(home.hass.config.path(), DOMAIN)
        await aiofiles.os.makedirs(storage_path, exist_ok=True)

        dash_path = os.path.join(storage_path, "quiet_solar_dashboard.yaml")
        async with aiofiles.open(dash_path, "w") as f_out:
            await f_out.write(rendered)

        dash_path = os.path.join(storage_path, "quiet_solar_dashboard_standard.yaml")
        async with aiofiles.open(dash_path, "w") as f_out:
            await f_out.write(rendered_std)

        storage_path_resources_destination: str = os.path.join(home.hass.config.path(), "www", DOMAIN)
        await aiofiles.os.makedirs(storage_path_resources_destination, exist_ok=True)

        resources_from_path: str = os.path.join(os.path.dirname(__file__), "resources")
        namespace_resource = generate_dashboard_resource_namespace(home)
        qs_tag = generate_dashboard_resource_qs_tag(home)
        resources_handler = _get_resource_handler(home)
        await qs_copy_resources_dir_and_resources_register_recursive(home, resources_from_path, storage_path_resources_destination, namespace_resource, qs_tag, resources_handler)


    except TemplateError as err:
        _LOGGER.error("generate_dashboard_yaml template error exception %s %s", err, exc_info=True,
                      stack_info=True)
        raise err
    except Exception as err:
        _LOGGER.error(f"Error rendering template: {err}", exc_info=True, stack_info=True)
        raise err



async def qs_copy_resources_dir_and_resources_register_recursive(home, from_dir_path:str, to_dir_path: str, namespace_resource : str, qs_tag: str, resources_handler: ResourceStorageCollection | None):
    await aiofiles.os.makedirs(to_dir_path, exist_ok=True)
    for entry in await aiofiles.os.listdir(from_dir_path):
        source_path = os.path.join(from_dir_path, entry)
        destination_path = os.path.join(to_dir_path, entry)
        if await aiofiles.os.path.isfile(source_path):
            async with aiofiles.open(source_path, "rb") as src, aiofiles.open(destination_path, "wb") as dest:
                await dest.write(await src.read())
            if resources_handler is not None:
                await update_resource(home, resources_handler, f"{namespace_resource}/{entry}", f"{namespace_resource}/{entry}?qs_tag={qs_tag}")

        elif await aiofiles.os.path.isdir(source_path):
            await qs_copy_resources_dir_and_resources_register_recursive(home, source_path, destination_path, namespace_resource + f"/{entry}", qs_tag, resources_handler)





