import logging
import os
from typing import TYPE_CHECKING

import aiofiles.os
from homeassistant.helpers.template import Template
from homeassistant.exceptions import TemplateError

from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def generate_dashboard_yaml(home):

    base_dir = os.path.join(os.path.dirname(__file__))
    base_dir = os.path.join(base_dir, "quiet_solar_dashboard_template.yaml.j2")

    with open(base_dir, "r") as f:
        template_content = f.read()

    # IMPORTANT: pass hass to the Template constructor
    tpl = Template(template_content, home.hass)

    try:
        rendered = tpl.async_render(variables={"home": home})
        # or: rendered = tpl.render(variables={"foo": "bar"})  # if you are in sync code
        # _LOGGER.warning(f"Template rendered: {rendered}")

        storage_path: str = os.path.join(home.hass.config.path(), DOMAIN)
        await aiofiles.os.makedirs(storage_path, exist_ok=True)
        base_dir = os.path.join(storage_path, "quiet_solar_dashboard.yaml")
        with open(base_dir, "w") as f:
            f.write(rendered)

    except TemplateError as err:
        # handle bad templates / runtime errors
        raise err
    except Exception as err:
        _LOGGER.error(f"Error rendering template: {err}", exc_info=err)
        raise err




