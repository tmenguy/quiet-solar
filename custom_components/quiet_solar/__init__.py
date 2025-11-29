from __future__ import annotations
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback, Event
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers import service
from homeassistant.components.persistent_notification import DOMAIN as PN_DOMAIN

from .data_handler import QSDataHandler

from homeassistant.helpers import config_validation as cv

_LOGGER = logging.getLogger(__name__)

from .const import (
    DATA_DEVICE_IDS,
    DOMAIN,
    DATA_HANDLER, DEVICE_TYPE
)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Quiet Solar component"""
    hass.data[DOMAIN] = {}
    
    # Register reload service
    register_reload_service(hass)
    
    # Register OCPP notification listener
    register_ocpp_notification_listener(hass)
    
    return True


async def async_reload_quiet_solar(hass: HomeAssistant, except_for_entry_id=None):
    # Then reload the entire integration by getting all entries and reloading them
    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if except_for_entry_id is not None and except_for_entry_id == entry.entry_id:
            continue
        try:
            await hass.config_entries.async_unload(entry.entry_id)
        except Exception as e:
            # Log the error but continue with the next entry
            _LOGGER.error(f"Error unloading entry {entry.entry_id}: {e}", exc_info=True, stack_info=True)


    hass.data[DOMAIN] = {}

    for entry in entries:
        if except_for_entry_id is not None and except_for_entry_id == entry.entry_id:
            continue
        try:
            await hass.config_entries.async_reload(entry.entry_id)
        except Exception as e:
            # Log the error but continue with the next entry
            _LOGGER.error(f"Error reloading entry {entry.entry_id}: {e}", exc_info=True, stack_info=True)

@callback
def register_reload_service(hass: HomeAssistant) -> None:
    """Register reload service for Quiet Solar."""
    async def _reload_integration(call: ServiceCall) -> None:
        """Reload all Quiet Solar config entries."""
        await async_reload_quiet_solar(hass)
    
    service.async_register_admin_service(
        hass,
        DOMAIN,
        "reload",
        _reload_integration,
    )

@callback
def register_ocpp_notification_listener(hass: HomeAssistant) -> None:
    """Register listener for OCPP notifications via persistent_notification service."""
    
    @callback
    async def ocpp_notification_listener(event: Event) -> None:
        """Listen for service call events from OCPP integration."""
        try:
            # Check if this is a persistent_notification.create service call
            if (event.data.get("domain") == PN_DOMAIN and 
                event.data.get("service") == "create"):

                _LOGGER.info(f"Received OCPP notification event: {event.data}")
                
                service_data = event.data.get("service_data", {})
                title = service_data.get("title", "")
                message = service_data.get("message", "")
                
                # Check if this is a notification from OCPP integration
                if "ocpp" in title.lower() or "charger" in message.lower():
                    _LOGGER.info(f"Intercepted OCPP notification: {title} - {message}")
                    
                    # Get the QSDataHandler to access chargers
                    data_handler = hass.data.get(DOMAIN, {}).get(DATA_HANDLER)
                    if not data_handler or not data_handler.home:
                        _LOGGER.debug("No QSDataHandler or home available for OCPP notification forwarding")
                        return
                    
                    # Find OCPP chargers using isinstance
                    from .ha_model.charger import QSChargerOCPP
                    ocpp_chargers = []
                    for charger in data_handler.home._chargers:
                        if isinstance(charger, QSChargerOCPP):
                            ocpp_chargers.append(charger)
                    
                    if ocpp_chargers:
                        matched_charger = False
                        for qs_charger in ocpp_chargers:
                            # Try to match the notification to the specific charger
                            if await _is_notification_for_charger(hass, message, qs_charger):
                                await qs_charger.handle_ocpp_notification(message, title)
                                matched_charger = True
                                break
                        
                        # If no specific match, send to all OCPP chargers
                        if not matched_charger:
                            for qs_charger in ocpp_chargers:
                                await qs_charger.handle_ocpp_notification(message, title)
                        
        except Exception as e:
            _LOGGER.error(f"Error in OCPP notification listener: {e}", exc_info=True, stack_info=True)
    
    # Listen for service call events
    hass.bus.async_listen("call_service", ocpp_notification_listener)

async def _is_notification_for_charger(hass: HomeAssistant, message: str, qs_charger) -> bool:
    """Check if a notification message is for a specific charger."""
    try:
        if not hasattr(qs_charger, 'charger_device_ocpp') or qs_charger.charger_device_ocpp is None:
            return False
            
        from homeassistant.helpers import device_registry
        device_reg = device_registry.async_get(hass)
        device = device_reg.async_get(qs_charger.charger_device_ocpp)
        
        if device is None:
            return False
            
        # Check if the message contains the charger ID or device name
        device_name = device.name_by_user or device.name or ""
        
        # Extract potential charger IDs from device identifiers
        charger_ids = []
        for identifier_domain, identifier_value in device.identifiers:
            if identifier_domain == "ocpp":
                charger_ids.append(str(identifier_value))
        
        # Check if message contains any of the charger identifiers or device name
        message_lower = message.lower()
        if device_name.lower() in message_lower:
            return True
            
        for charger_id in charger_ids:
            if charger_id.lower() in message_lower:
                return True
        
        # Also check the charger name from quiet solar
        if qs_charger.name and qs_charger.name.lower() in message_lower:
            return True
                
        return False
        
    except Exception as e:
        _LOGGER.debug(f"Error matching notification to charger: {e}", exc_info=True, stack_info=True)
        return False

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Quiet Solar integration."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler is None:
        data_handler = QSDataHandler(hass)
        hass.data[DOMAIN][DATA_HANDLER] = data_handler

    await data_handler.async_add_entry(entry)
    
    # Register update listener to reload the entry when config entry is updated
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    data_handler = hass.data[DOMAIN].get(DATA_HANDLER)
    if data_handler and entry.entry_id in hass.data[DOMAIN]:
        device = hass.data[DOMAIN][entry.entry_id]
        platforms = device.get_platforms()

        if device.home:
            device.home.remove_device(device)

        if platforms:
            unload_ok = await hass.config_entries.async_unload_platforms(entry, platforms)
            if unload_ok:
                hass.data[DOMAIN].pop(entry.entry_id)
                return True
    
    return False

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry."""
    await hass.config_entries.async_reload(entry.entry_id)

