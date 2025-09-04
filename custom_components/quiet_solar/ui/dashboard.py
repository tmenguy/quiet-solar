import logging
import os
from typing import TYPE_CHECKING


import aiofiles.os
import aiofiles
from homeassistant.helpers.template import Template
from homeassistant.exceptions import TemplateError

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def generate_dashboard_yaml(home):
    base_dir = os.path.join(os.path.dirname(__file__))
    jinja_path = os.path.join(base_dir, "quiet_solar_dashboard_template.yaml.j2")

    _LOGGER.warning("generate_dashboard_yaml: regenerating dashboard YAML file")

    # Async, non-blocking read of the Jinja template
    async with aiofiles.open(jinja_path, "r") as f:
        template_content = await f.read()

    tpl = Template(template_content, home.hass)

    try:
        rendered = tpl.async_render(variables={"home": home})

        storage_path: str = os.path.join(home.hass.config.path(), DOMAIN)
        await aiofiles.os.makedirs(storage_path, exist_ok=True)
        base_dir = os.path.join(storage_path, "quiet_solar_dashboard.yaml")
        async with aiofiles.open(base_dir, "w") as f_out:
            await f_out.write(rendered)

        storage_path_resources_destintion: str = os.path.join(home.hass.config.path(), "www", DOMAIN)
        await aiofiles.os.makedirs(storage_path_resources_destintion, exist_ok=True)

        resources_from_path: str = os.path.join(os.path.dirname(__file__), "resources")

        for entry in await aiofiles.os.listdir(resources_from_path):
            source_path = os.path.join(resources_from_path, entry)
            destination_path = os.path.join(storage_path_resources_destintion, entry)

            if await aiofiles.os.path.isfile(source_path):
                async with aiofiles.open(source_path, "rb") as src, aiofiles.open(destination_path, "wb") as dest:
                    await dest.write(await src.read())
            elif await aiofiles.os.path.isdir(source_path):
                await aiofiles.os.makedirs(destination_path, exist_ok=True)
                for root, dirs, files in os.walk(source_path):
                    for dir_ in dirs:
                        await aiofiles.os.makedirs(
                            os.path.join(destination_path, os.path.relpath(os.path.join(root, dir_), source_path)),
                            exist_ok=True
                        )
                    for file_ in files:
                        src_file = os.path.join(root, file_)
                        dest_file = os.path.join(destination_path, os.path.relpath(src_file, source_path))
                        async with aiofiles.open(src_file, "rb") as src, aiofiles.open(dest_file, "wb") as dest:
                            await dest.write(await src.read())
    except TemplateError as err:
        raise err
    except Exception as err:
        _LOGGER.error(f"Error rendering template: {err}", exc_info=err)
        raise err




