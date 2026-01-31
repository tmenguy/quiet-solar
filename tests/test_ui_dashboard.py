"""Tests for the dashboard module."""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from tests.factories import create_minimal_home_model
from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.ui.dashboard import (
    generate_dashboard_resource_qs_tag,
    generate_dashboard_resource_namespace,
    _get_resource_handler,
    update_resource,
    generate_dashboard_yaml,
    qs_copy_resources_dir_and_resources_register_recursive,
)


class MockResourceStorageCollection:
    """Mock ResourceStorageCollection for testing."""

    def __init__(self, items: List[Dict[str, Any]] = None):
        self._items = items or []
        self.loaded = False
        self.created_items = []
        self.updated_items = []
        self.store = MagicMock()
        self.store.key = "lovelace_resources"
        self.store.version = 1

    async def async_load(self):
        """Load the resources."""
        self.loaded = True

    def async_items(self) -> List[Dict[str, Any]]:
        """Return all items."""
        return self._items

    async def async_create_item(self, data: Dict[str, Any]):
        """Create a new item."""
        self.created_items.append(data)

    async def async_update_item(self, item_id: str, data: Dict[str, Any]):
        """Update an existing item."""
        self.updated_items.append((item_id, data))


class MockHome:
    """Mock home object for testing."""

    def __init__(self, hass, dashboard_sections=None):
        self.hass = hass
        self.name = "Test Home"
        self.device_type = "home"
        self.dashboard_sections = dashboard_sections or [("Main", None)]
        self.devices = []
        self.ha_entities = {}

    def get_devices_for_dashboard_section(self, section_name: str):
        """Return devices for a dashboard section."""
        return self.devices


def create_mock_hass(config_dir: str = None) -> MagicMock:
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}

    config = MagicMock()
    config_dir = config_dir or tempfile.mkdtemp()
    config.path = MagicMock(side_effect=lambda *args: os.path.join(config_dir, *args) if args else config_dir)
    hass.config = config

    return hass


# =============================================================================
# Tests for generate_dashboard_resource_qs_tag
# =============================================================================

def test_generate_dashboard_resource_qs_tag_returns_string():
    """Test that qs_tag is a string."""
    mock_home = create_minimal_home_model()
    tag = generate_dashboard_resource_qs_tag(mock_home)
    assert isinstance(tag, str)


def test_generate_dashboard_resource_qs_tag_is_numeric():
    """Test that qs_tag is a numeric string (epoch seconds)."""
    mock_home = create_minimal_home_model()
    tag = generate_dashboard_resource_qs_tag(mock_home)
    assert tag.isdigit()


def test_generate_dashboard_resource_qs_tag_changes_over_time():
    """Test that qs_tag changes between calls (with time mocking)."""
    import time
    mock_home = create_minimal_home_model()

    with patch.object(time, 'time', return_value=1000):
        tag1 = generate_dashboard_resource_qs_tag(mock_home)

    with patch.object(time, 'time', return_value=2000):
        tag2 = generate_dashboard_resource_qs_tag(mock_home)

    assert tag1 == "1000"
    assert tag2 == "2000"
    assert tag1 != tag2


# =============================================================================
# Tests for generate_dashboard_resource_namespace
# =============================================================================

def test_generate_dashboard_resource_namespace_format():
    """Test that namespace follows the correct format."""
    mock_home = create_minimal_home_model()
    namespace = generate_dashboard_resource_namespace(mock_home)
    assert namespace == f"/local/{DOMAIN}"


def test_generate_dashboard_resource_namespace_starts_with_local():
    """Test that namespace starts with /local/."""
    mock_home = create_minimal_home_model()
    namespace = generate_dashboard_resource_namespace(mock_home)
    assert namespace.startswith("/local/")


# =============================================================================
# Tests for _get_resource_handler
# =============================================================================

def test_get_resource_handler_no_hass_data():
    """Test when hass.data is empty/None."""
    mock_home = create_minimal_home_model()
    mock_home.hass.data = None

    result = _get_resource_handler(mock_home)
    assert result is None


def test_get_resource_handler_no_lovelace_data():
    """Test when lovelace data is missing."""
    mock_home = create_minimal_home_model()
    mock_home.hass.data = {}

    result = _get_resource_handler(mock_home)
    assert result is None


def test_get_resource_handler_no_resources_in_lovelace():
    """Test when lovelace has no resources (old HA version)."""
    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {}}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_no_resources_in_lovelace_new_ha():
    """Test when lovelace has no resources (new HA version)."""
    mock_lovelace = MagicMock()
    mock_lovelace.resources = None

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": mock_lovelace}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=True)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_yaml_mode_no_store():
    """Test when resources has no store (YAML mode)."""
    mock_resources = MagicMock()
    mock_resources.store = None

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {"resources": mock_resources}}

    # Test with version < 2025.2.0
    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_yaml_mode_no_store_attr():
    """Test when resources doesn't have store attribute (YAML mode)."""
    mock_resources = MagicMock(spec=[])  # No attributes

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {"resources": mock_resources}}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_wrong_store_key():
    """Test when store has wrong key."""
    mock_resources = MagicMock()
    mock_resources.store.key = "wrong_key"
    mock_resources.store.version = 1

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {"resources": mock_resources}}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_wrong_store_version():
    """Test when store has wrong version."""
    mock_resources = MagicMock()
    mock_resources.store.key = "lovelace_resources"
    mock_resources.store.version = 2  # Wrong version

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {"resources": mock_resources}}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result is None


def test_get_resource_handler_success_old_ha_version():
    """Test successful resource handler retrieval for HA < 2025.2."""
    mock_resources = MagicMock()
    mock_resources.store.key = "lovelace_resources"
    mock_resources.store.version = 1

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": {"resources": mock_resources}}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=False)
        result = _get_resource_handler(mock_home)

    assert result == mock_resources


def test_get_resource_handler_success_new_ha_version():
    """Test successful resource handler retrieval for HA >= 2025.2."""
    mock_resources = MagicMock()
    mock_resources.store.key = "lovelace_resources"
    mock_resources.store.version = 1

    mock_lovelace = MagicMock()
    mock_lovelace.resources = mock_resources

    mock_home = create_minimal_home_model()
    mock_home.hass.data = {"lovelace": mock_lovelace}

    with patch('custom_components.quiet_solar.ui.dashboard.AwesomeVersion') as mock_av:
        mock_av.return_value.__gt__ = MagicMock(return_value=True)
        result = _get_resource_handler(mock_home)

    assert result == mock_resources


# =============================================================================
# Tests for update_resource
# =============================================================================

@pytest.mark.asyncio
async def test_update_resource_loads_if_not_loaded():
    """Test that resources are loaded if not already loaded."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection()

    await update_resource(mock_home, resources, "/local/test", "/local/test?qs_tag=123")

    assert resources.loaded is True


@pytest.mark.asyncio
async def test_update_resource_skips_load_if_already_loaded():
    """Test that loading is skipped if already loaded."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection()
    resources.loaded = True

    # Mock async_load to track calls
    original_load = resources.async_load
    load_called = False

    async def track_load():
        nonlocal load_called
        load_called = True
        await original_load()

    resources.async_load = track_load

    await update_resource(mock_home, resources, "/local/test", "/local/test?qs_tag=123")

    assert load_called is False


@pytest.mark.asyncio
async def test_update_resource_creates_new_resource():
    """Test creating a new resource when none exists."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection()

    await update_resource(mock_home, resources, "/local/test", "/local/test?qs_tag=123")

    assert len(resources.created_items) == 1
    assert resources.created_items[0]["res_type"] == "module"
    assert resources.created_items[0]["url"] == "/local/test?qs_tag=123"


@pytest.mark.asyncio
async def test_update_resource_updates_existing_resource():
    """Test updating an existing resource."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection([
        {"id": "resource_1", "url": "/local/test?qs_tag=old"},
    ])

    await update_resource(mock_home, resources, "/local/test", "/local/test?qs_tag=new")

    assert len(resources.updated_items) == 1
    assert resources.updated_items[0][0] == "resource_1"
    assert resources.updated_items[0][1]["url"] == "/local/test?qs_tag=new"


@pytest.mark.asyncio
async def test_update_resource_no_update_if_same_url():
    """Test that no update occurs if URL is the same."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection([
        {"id": "resource_1", "url": "/local/test?qs_tag=123"},
    ])

    await update_resource(mock_home, resources, "/local/test", "/local/test?qs_tag=123")

    assert len(resources.updated_items) == 0
    assert len(resources.created_items) == 0


@pytest.mark.asyncio
async def test_update_resource_matches_by_prefix():
    """Test that resources are matched by URL prefix."""
    mock_home = create_minimal_home_model()
    resources = MockResourceStorageCollection([
        {"id": "resource_1", "url": "/local/other/file.js?tag=1"},
        {"id": "resource_2", "url": "/local/test/file.js?tag=old"},
    ])

    await update_resource(mock_home, resources, "/local/test/file.js", "/local/test/file.js?qs_tag=new")

    assert len(resources.updated_items) == 1
    assert resources.updated_items[0][0] == "resource_2"


# =============================================================================
# Tests for generate_dashboard_yaml
# =============================================================================

@pytest.mark.asyncio
async def test_generate_dashboard_yaml_creates_directories():
    """Test that generate_dashboard_yaml creates necessary directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        # Mock the template rendering
        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.return_value = "# Test YAML"
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.qs_copy_resources_dir_and_resources_register_recursive', new_callable=AsyncMock):
                    await generate_dashboard_yaml(mock_home)

        # Check that directories were created
        assert os.path.exists(os.path.join(tmp_dir, DOMAIN))
        assert os.path.exists(os.path.join(tmp_dir, "www", DOMAIN))


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_writes_files():
    """Test that generate_dashboard_yaml writes dashboard YAML files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        rendered_content = "# Rendered Dashboard YAML"

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.return_value = rendered_content
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.qs_copy_resources_dir_and_resources_register_recursive', new_callable=AsyncMock):
                    await generate_dashboard_yaml(mock_home)

        # Check that files were written
        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")
        dash_path_standard = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard_standard.yaml")

        assert os.path.exists(dash_path)
        assert os.path.exists(dash_path_standard)

        with open(dash_path, "r") as f:
            assert f.read() == rendered_content


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_passes_home_to_template():
    """Test that home object is passed to the template."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        captured_variables = []

        def capture_render(variables):
            captured_variables.append(variables)
            return "# YAML"

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.side_effect = capture_render
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.qs_copy_resources_dir_and_resources_register_recursive', new_callable=AsyncMock):
                    await generate_dashboard_yaml(mock_home)

        assert len(captured_variables) == 2
        assert captured_variables[0]["home"] is mock_home
        assert captured_variables[1]["home"] is mock_home


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_handles_template_error():
    """Test that template errors are properly raised."""
    from homeassistant.exceptions import TemplateError

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.side_effect = TemplateError("Test template error")
            mock_template_cls.return_value = mock_template

            with pytest.raises(TemplateError):
                await generate_dashboard_yaml(mock_home)


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_handles_generic_error():
    """Test that generic errors are properly raised."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.side_effect = RuntimeError("Generic error")
            mock_template_cls.return_value = mock_template

            with pytest.raises(RuntimeError):
                await generate_dashboard_yaml(mock_home)


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_copies_resources():
    """Test that resources are copied."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        copy_called_with = []

        async def capture_copy(*args):
            copy_called_with.append(args)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.return_value = "# YAML"
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.qs_copy_resources_dir_and_resources_register_recursive', side_effect=capture_copy):
                    await generate_dashboard_yaml(mock_home)

        assert len(copy_called_with) == 1
        # Check that the source is the resources directory
        assert copy_called_with[0][1].endswith("resources")
        # Check that the destination is www/quiet_solar
        assert copy_called_with[0][2].endswith(os.path.join("www", DOMAIN))


# =============================================================================
# Tests for qs_copy_resources_dir_and_resources_register_recursive
# =============================================================================

@pytest.mark.asyncio
async def test_qs_copy_resources_creates_destination_dir():
    """Test that destination directory is created."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        mock_home = create_minimal_home_model()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", None
        )

        assert os.path.exists(to_dir)


@pytest.mark.asyncio
async def test_qs_copy_resources_copies_files():
    """Test that files are copied from source to destination."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        # Create a test file
        test_file = os.path.join(from_dir, "test.js")
        with open(test_file, "w") as f:
            f.write("console.log('test');")

        mock_home = create_minimal_home_model()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", None
        )

        dest_file = os.path.join(to_dir, "test.js")
        assert os.path.exists(dest_file)
        with open(dest_file, "r") as f:
            assert f.read() == "console.log('test');"


@pytest.mark.asyncio
async def test_qs_copy_resources_copies_binary_files():
    """Test that binary files are copied correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        # Create a binary test file
        test_file = os.path.join(from_dir, "test.bin")
        binary_content = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD])
        with open(test_file, "wb") as f:
            f.write(binary_content)

        mock_home = create_minimal_home_model()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", None
        )

        dest_file = os.path.join(to_dir, "test.bin")
        assert os.path.exists(dest_file)
        with open(dest_file, "rb") as f:
            assert f.read() == binary_content


@pytest.mark.asyncio
async def test_qs_copy_resources_recursive_directories():
    """Test that subdirectories are copied recursively."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        subdir = os.path.join(from_dir, "subdir")
        os.makedirs(subdir)

        # Create files in root and subdirectory
        root_file = os.path.join(from_dir, "root.js")
        with open(root_file, "w") as f:
            f.write("root")

        sub_file = os.path.join(subdir, "sub.js")
        with open(sub_file, "w") as f:
            f.write("sub")

        mock_home = create_minimal_home_model()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", None
        )

        assert os.path.exists(os.path.join(to_dir, "root.js"))
        assert os.path.exists(os.path.join(to_dir, "subdir", "sub.js"))


@pytest.mark.asyncio
async def test_qs_copy_resources_registers_resources():
    """Test that resources are registered with resource handler."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        # Create test files
        with open(os.path.join(from_dir, "test1.js"), "w") as f:
            f.write("test1")
        with open(os.path.join(from_dir, "test2.js"), "w") as f:
            f.write("test2")

        mock_home = create_minimal_home_model()
        resources = MockResourceStorageCollection()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", resources
        )

        # Check that resources were registered
        assert len(resources.created_items) == 2


@pytest.mark.asyncio
async def test_qs_copy_resources_registers_with_qs_tag():
    """Test that resources are registered with qs_tag parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        with open(os.path.join(from_dir, "test.js"), "w") as f:
            f.write("test")

        mock_home = create_minimal_home_model()
        resources = MockResourceStorageCollection()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "12345", resources
        )

        assert len(resources.created_items) == 1
        assert resources.created_items[0]["url"] == "/local/test/test.js?qs_tag=12345"


@pytest.mark.asyncio
async def test_qs_copy_resources_no_register_when_handler_none():
    """Test that nothing is registered when handler is None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        with open(os.path.join(from_dir, "test.js"), "w") as f:
            f.write("test")

        mock_home = create_minimal_home_model()

        # Should not raise, just skip registration
        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", None
        )

        # File should still be copied
        assert os.path.exists(os.path.join(to_dir, "test.js"))


@pytest.mark.asyncio
async def test_qs_copy_resources_recursive_namespace():
    """Test that namespace is updated correctly for subdirectories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        subdir = os.path.join(from_dir, "cards")
        os.makedirs(subdir)

        with open(os.path.join(subdir, "card.js"), "w") as f:
            f.write("card")

        mock_home = create_minimal_home_model()
        resources = MockResourceStorageCollection()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", resources
        )

        # Check that the URL includes the subdirectory
        assert len(resources.created_items) == 1
        assert resources.created_items[0]["url"] == "/local/test/cards/card.js?qs_tag=123"


@pytest.mark.asyncio
async def test_qs_copy_resources_empty_directory():
    """Test handling of empty source directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        mock_home = create_minimal_home_model()
        resources = MockResourceStorageCollection()

        await qs_copy_resources_dir_and_resources_register_recursive(
            mock_home, from_dir, to_dir, "/local/test", "123", resources
        )

        assert os.path.exists(to_dir)
        assert len(resources.created_items) == 0


# =============================================================================
# Integration-like tests
# =============================================================================

@pytest.mark.asyncio
async def test_full_dashboard_generation_flow():
    """Test the complete dashboard generation flow."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        # Create a mock resources directory structure
        ui_dir = os.path.join(tmp_dir, "ui_resources")
        os.makedirs(ui_dir)
        with open(os.path.join(ui_dir, "card.js"), "w") as f:
            f.write("// Custom card")

        # Run the flow
        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.return_value = "views:\n  - type: sections"
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.os.path.dirname') as mock_dirname:
                    mock_dirname.return_value = tmp_dir
                    # Create the expected template files
                    with open(os.path.join(tmp_dir, "quiet_solar_dashboard_template.yaml.j2"), "w") as f:
                        f.write("{{ home.name }}")
                    with open(os.path.join(tmp_dir, "quiet_solar_dashboard_template_standard_ha.yaml.j2"), "w") as f:
                        f.write("{{ home.name }}")

                    os.makedirs(os.path.join(tmp_dir, "resources"))
                    with open(os.path.join(tmp_dir, "resources", "test.js"), "w") as f:
                        f.write("// test")

                    await generate_dashboard_yaml(mock_home)

        # Verify outputs
        domain_dir = os.path.join(tmp_dir, DOMAIN)
        assert os.path.exists(domain_dir)
        assert os.path.exists(os.path.join(domain_dir, "quiet_solar_dashboard.yaml"))
        assert os.path.exists(os.path.join(domain_dir, "quiet_solar_dashboard_standard.yaml"))


@pytest.mark.asyncio
async def test_dashboard_generation_with_resource_registration():
    """Test dashboard generation with resource registration enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        resources = MockResourceStorageCollection()

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_template_cls:
            mock_template = MagicMock()
            mock_template.async_render.return_value = "# Dashboard"
            mock_template_cls.return_value = mock_template

            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=resources):
                with patch('custom_components.quiet_solar.ui.dashboard.os.path.dirname') as mock_dirname:
                    mock_dirname.return_value = tmp_dir

                    # Create template files
                    with open(os.path.join(tmp_dir, "quiet_solar_dashboard_template.yaml.j2"), "w") as f:
                        f.write("{{ home.name }}")
                    with open(os.path.join(tmp_dir, "quiet_solar_dashboard_template_standard_ha.yaml.j2"), "w") as f:
                        f.write("{{ home.name }}")

                    # Create resources
                    os.makedirs(os.path.join(tmp_dir, "resources"))
                    with open(os.path.join(tmp_dir, "resources", "qs-card.js"), "w") as f:
                        f.write("// QS card")

                    await generate_dashboard_yaml(mock_home)

        # Verify resource was registered
        assert len(resources.created_items) == 1
        assert "qs-card.js" in resources.created_items[0]["url"]


# =============================================================================
# Real Template Rendering Tests (No Mocking)
# =============================================================================

class MockEntity:
    """Mock entity with entity_id and name."""
    def __init__(self, entity_id, name):
        self.entity_id = entity_id
        self.name = name


class RealDeviceMock:
    """More complete device mock for real template rendering."""

    def __init__(self, device_type, name, dashboard_section="Main"):
        self.device_type = device_type
        self.name = name
        self.device_id = f"mock_{device_type}_{name}"
        self.dashboard_section = dashboard_section
        self.dashboard_sort_string = name
        self.ha_entities = {}
        self.calendar = None
        self.car_tracker = None
        self.car_is_invited = False
        # Additional car-specific attributes needed by templates
        self.car_charge_percent_sensor = None

    def get_attached_virtual_devices(self):
        """Return empty list for virtual devices."""
        return []

    def can_use_charge_percent_constraints(self):
        """For car devices."""
        return True


def create_real_home_for_dashboard(hass, dashboard_sections=None):
    """Create a more realistic home object for dashboard testing."""
    home = MockHome(hass, dashboard_sections or [("Main", "mdi:home"), ("Cars", "mdi:car")])

    # Add some devices to different sections
    car = RealDeviceMock("car", "Test Car", "Cars")
    car.car_tracker = "device_tracker.test_car"
    car.car_charge_percent_sensor = "sensor.test_car_soc"
    car.ha_entities = {
        "selected_charger_for_car": MockEntity("select.test_car_charger", "Charger"),
        "car_soc_percentage": MockEntity("sensor.test_car_soc", "Battery Level"),
        "best_power_value": MockEntity("sensor.test_car_power", "Charging Power"),
        "car_charge_type": MockEntity("sensor.test_car_charge_type", "Charge Type"),
        "car_estimated_range_km": MockEntity("sensor.test_car_range", "Range"),
    }

    climate = RealDeviceMock("climate", "Heat Pump", "Main")
    climate.ha_entities = {
        "qs_enable_device": MockEntity("switch.heat_pump_enable", "Enable Device"),
        "qs_best_effort_green_only": MockEntity("switch.heat_pump_green_only", "Green Only"),
        "best_power_value": MockEntity("sensor.heat_pump_power", "Power"),
        "load_current_command": MockEntity("sensor.heat_pump_command", "Current Command"),
    }

    pool = RealDeviceMock("pool", "Pool Pump", "Main")
    pool.ha_entities = {
        "qs_enable_device": MockEntity("switch.pool_enable", "Enable Pool"),
        "default_on_duration": MockEntity("number.pool_duration", "Run Duration"),
        "best_power_value": MockEntity("sensor.pool_power", "Power"),
    }

    battery = RealDeviceMock("battery", "Home Battery", "Main")
    battery.ha_entities = {
        "load_current_command": MockEntity("sensor.battery_command", "Battery Command"),
    }

    person = RealDeviceMock("person", "John Doe", "Main")
    person.ha_entities = {
        "person_mileage_prediction": MockEntity("sensor.john_mileage_prediction", "Mileage Prediction"),
    }

    home_device = RealDeviceMock("home", home.name, "Main")
    home_device.ha_entities = {
        "home_mode": MockEntity("select.home_mode", "Home Mode"),
        "qs_home_is_off_grid": MockEntity("switch.home_off_grid", "Off Grid Mode"),
        "qs_home_serialize_for_debug": MockEntity("button.home_debug", "Debug"),
        "qs_home_generate_yaml_dashboard": MockEntity("button.home_generate_dashboard", "Generate Dashboard"),
    }

    home.devices = [car, climate, pool, battery, person, home_device]

    return home


@pytest.mark.asyncio
async def test_real_dashboard_yaml_generation_no_mocking():
    """Test actual dashboard YAML generation with real Jinja template rendering."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass)

        # Copy the actual template files from the source
        import shutil
        from custom_components.quiet_solar.ui import dashboard
        ui_dir = os.path.dirname(dashboard.__file__)

        # Use the real templates
        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        # Verify files were created
        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")
        dash_path_standard = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard_standard.yaml")

        assert os.path.exists(dash_path), "Main dashboard file was not created"
        assert os.path.exists(dash_path_standard), "Standard dashboard file was not created"

        # Read and verify the generated YAML files are not empty
        with open(dash_path, "r") as f:
            content = f.read()

        assert len(content) > 0, "Main dashboard YAML is empty"
        assert "views:" in content, "Main dashboard YAML doesn't contain 'views:'"
        print(f"\nMain dashboard YAML length: {len(content)} characters")

        with open(dash_path_standard, "r") as f:
            content_standard = f.read()

        assert len(content_standard) > 0, "Standard dashboard YAML is empty"
        assert "views:" in content_standard, "Standard dashboard YAML doesn't contain 'views:'"
        print(f"Standard dashboard YAML length: {len(content_standard)} characters")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_contains_devices():
    """Test that generated dashboard YAML actually contains device information."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Check for section names
        assert "Main" in content, "Main section not found in dashboard"
        assert "Cars" in content, "Cars section not found in dashboard"

        # Check for device names
        assert "Test Car" in content, "Test Car device not found in dashboard"
        assert "Heat Pump" in content, "Heat Pump device not found in dashboard"
        assert "Pool Pump" in content, "Pool Pump device not found in dashboard"
        assert "Home Battery" in content, "Home Battery device not found in dashboard"
        assert "John Doe" in content, "John Doe person not found in dashboard"

        # Check for entity IDs
        assert "sensor.test_car_soc" in content, "Car SoC entity not found"
        assert "switch.heat_pump_enable" in content, "Heat pump enable switch not found"
        assert "sensor.battery_command" in content, "Battery command sensor not found"
        assert "sensor.john_mileage_prediction" in content, "Person mileage prediction sensor not found"

        print(f"\n✓ Dashboard contains all expected devices and entities (including battery and person)")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_valid_structure():
    """Test that generated dashboard YAML has valid structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Parse as YAML to verify it's valid
        import yaml
        try:
            dashboard_data = yaml.safe_load(content)
            assert dashboard_data is not None, "Dashboard YAML parsed to None"
            assert "views" in dashboard_data, "Dashboard doesn't have 'views' key"
            assert isinstance(dashboard_data["views"], list), "Views is not a list"
            assert len(dashboard_data["views"]) > 0, "No views generated"

            print(f"\n✓ Dashboard YAML is valid with {len(dashboard_data['views'])} views")

            # Check first view structure
            first_view = dashboard_data["views"][0]
            assert "type" in first_view, "View doesn't have type"
            assert first_view["type"] == "sections", "View type is not 'sections'"
            assert "title" in first_view, "View doesn't have title"
            assert "sections" in first_view, "View doesn't have sections"

            print(f"✓ First view: '{first_view['title']}' with {len(first_view['sections'])} sections")

        except yaml.YAMLError as e:
            pytest.fail(f"Generated dashboard YAML is invalid: {e}")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_car_custom_card():
    """Test that car devices use custom:qs-car-card type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Check for custom card type
        assert "custom:qs-car-card" in content, "Custom car card type not found in dashboard"

        # Parse and verify structure
        import yaml
        dashboard_data = yaml.safe_load(content)

        # Find the car view
        car_view = None
        for view in dashboard_data["views"]:
            if view["title"] == "Cars":
                car_view = view
                break

        assert car_view is not None, "Cars view not found"
        assert len(car_view["sections"]) > 0, "Cars view has no sections"

        # Check that the car card has the custom type
        car_section = car_view["sections"][0]
        assert "cards" in car_section, "Car section has no cards"
        car_card = car_section["cards"][0]
        assert car_card["type"] == "custom:qs-car-card", "Car card is not custom:qs-car-card"

        print(f"\n✓ Car uses custom:qs-car-card with title: {car_card['title']}")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_with_empty_sections():
    """Test dashboard generation when some sections have no devices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        # Create home with sections but assign devices only to one section
        mock_home = MockHome(mock_hass, [("Main", "mdi:home"), ("Empty", "mdi:ghost"), ("Cars", "mdi:car")])

        car = RealDeviceMock("car", "Test Car", "Cars")
        car.car_charge_percent_sensor = "sensor.test_car_soc"
        car.ha_entities = {
            "car_soc_percentage": MockEntity("sensor.test_car_soc", "Battery Level"),
        }
        mock_home.devices = [car]

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Parse YAML
        import yaml
        dashboard_data = yaml.safe_load(content)

        # The template includes empty sections with their titles - it doesn't exclude them
        # So we just verify Cars section is present
        view_titles = [view["title"] for view in dashboard_data["views"]]
        assert "Cars" in view_titles, "Cars section should be present"

        # Verify that only Cars section has actual device content
        cars_view = [v for v in dashboard_data["views"] if v["title"] == "Cars"][0]
        assert len(cars_view["sections"]) > 0, "Cars view should have sections with devices"

        print(f"\n✓ Dashboard generated with views: {view_titles}")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_multiple_device_types():
    """Test dashboard with multiple device types in same section."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass, [("Devices", "mdi:devices")])

        # Add various device types
        devices = []

        car = RealDeviceMock("car", "Electric Car", "Devices")
        car.car_charge_percent_sensor = "sensor.car_soc"
        car.ha_entities = {"car_soc_percentage": MockEntity("sensor.car_soc", "SoC")}
        devices.append(car)

        climate = RealDeviceMock("climate", "Heat Pump", "Devices")
        climate.ha_entities = {"qs_enable_device": MockEntity("switch.hp_enable", "Enable")}
        devices.append(climate)

        pool = RealDeviceMock("pool", "Pool Pump", "Devices")
        pool.ha_entities = {"qs_enable_device": MockEntity("switch.pool_enable", "Enable")}
        devices.append(pool)

        on_off = RealDeviceMock("on_off_duration", "Washing Machine", "Devices")
        on_off.ha_entities = {"qs_enable_device": MockEntity("switch.washer_enable", "Enable")}
        devices.append(on_off)

        battery = RealDeviceMock("battery", "Home Battery", "Devices")
        battery.ha_entities = {"load_current_command": MockEntity("sensor.battery_cmd", "Command")}
        devices.append(battery)

        person = RealDeviceMock("person", "Jane Smith", "Devices")
        person.ha_entities = {"person_mileage_prediction": MockEntity("sensor.jane_mileage", "Mileage")}
        devices.append(person)

        home_device = RealDeviceMock("home", "Home Control", "Devices")
        home_device.ha_entities = {"home_mode": MockEntity("select.home_mode", "Mode")}
        devices.append(home_device)

        mock_home.devices = devices

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Parse YAML
        import yaml
        dashboard_data = yaml.safe_load(content)

        assert len(dashboard_data["views"]) == 1, "Should have exactly one view"
        view = dashboard_data["views"][0]
        assert view["title"] == "Devices"
        assert len(view["sections"]) == 7, f"Should have 7 sections (one per device), got {len(view['sections'])}"

        # Verify each device appears
        section_titles = [section["cards"][0]["title"] for section in view["sections"]]
        assert "Electric Car" in section_titles
        assert "Heat Pump" in section_titles
        assert "Pool Pump" in section_titles
        assert "Washing Machine" in section_titles
        assert "Home Battery" in section_titles
        assert "Jane Smith" in section_titles
        assert "Home Control" in section_titles

        print(f"\n✓ All {len(devices)} device types correctly rendered in dashboard (including battery and person)")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_file_sizes():
    """Test that generated dashboard files have reasonable sizes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")
        dash_path_standard = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard_standard.yaml")

        # Check file sizes
        main_size = os.path.getsize(dash_path)
        standard_size = os.path.getsize(dash_path_standard)

        # Files should be at least 100 bytes (not empty or just a few chars)
        assert main_size > 100, f"Main dashboard too small: {main_size} bytes"
        assert standard_size > 100, f"Standard dashboard too small: {standard_size} bytes"

        # Files should not be unreasonably large (> 1MB would be suspicious)
        assert main_size < 1_000_000, f"Main dashboard too large: {main_size} bytes"
        assert standard_size < 1_000_000, f"Standard dashboard too large: {standard_size} bytes"

        print(f"\n✓ Main dashboard: {main_size} bytes")
        print(f"✓ Standard dashboard: {standard_size} bytes")


@pytest.mark.asyncio
async def test_real_dashboard_yaml_battery_and_person_specific():
    """Test that battery and person devices render with correct entities."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass, [("Energy", "mdi:lightning-bolt")])

        # Add battery and person devices
        battery = RealDeviceMock("battery", "Tesla Powerwall", "Energy")
        battery.ha_entities = {
            "load_current_command": MockEntity("sensor.powerwall_command", "Battery Command"),
        }

        person = RealDeviceMock("person", "Alice Johnson", "Energy")
        person.ha_entities = {
            "person_mileage_prediction": MockEntity("sensor.alice_mileage_prediction", "Alice Mileage"),
        }

        mock_home.devices = [battery, person]

        with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
            await generate_dashboard_yaml(mock_home)

        dash_path = os.path.join(tmp_dir, DOMAIN, "quiet_solar_dashboard.yaml")

        with open(dash_path, "r") as f:
            content = f.read()

        # Parse YAML to verify structure
        import yaml
        dashboard_data = yaml.safe_load(content)

        # Should have one view (Energy)
        assert len(dashboard_data["views"]) == 1
        view = dashboard_data["views"][0]
        assert view["title"] == "Energy"

        # Should have 2 sections (battery and person)
        assert len(view["sections"]) == 2

        # Check battery content
        assert "Tesla Powerwall" in content
        assert "sensor.powerwall_command" in content

        # Check person content
        assert "Alice Johnson" in content
        assert "sensor.alice_mileage_prediction" in content

        # Verify the entity structure in YAML
        battery_section = None
        person_section = None
        for section in view["sections"]:
            title = section["cards"][0]["title"]
            if title == "Tesla Powerwall":
                battery_section = section
            elif title == "Alice Johnson":
                person_section = section

        assert battery_section is not None, "Battery section not found"
        assert person_section is not None, "Person section not found"

        # Check battery entities
        battery_entities = battery_section["cards"][0]["entities"]
        battery_entity_ids = [e["entity"] for e in battery_entities if isinstance(e, dict)]
        assert "sensor.powerwall_command" in battery_entity_ids

        # Check person entities
        person_entities = person_section["cards"][0]["entities"]
        person_entity_ids = [e["entity"] for e in person_entities if isinstance(e, dict)]
        assert "sensor.alice_mileage_prediction" in person_entity_ids

        print(f"\n✓ Battery device rendered with load_current_command sensor")
        print(f"✓ Person device rendered with person_mileage_prediction sensor")



