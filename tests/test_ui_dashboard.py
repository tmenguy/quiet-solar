"""Tests for the dashboard module."""
from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

from tests.factories import create_minimal_home_model
from custom_components.quiet_solar.const import DOMAIN
from custom_components.quiet_solar.ui.dashboard import (
    generate_dashboard_resource_qs_tag,
    generate_dashboard_resource_namespace,
    _get_resource_handler,
    update_resource,
    generate_dashboard_yaml,
    _async_copy_and_register_resources,
    async_restore_dashboards_and_update_resources,
    async_unregister_dashboards,
    async_auto_generate_if_first_install,
    async_update_resources,
    _async_register_or_update_dashboard,
    _async_save_dashboard_tracking,
    _async_load_tracking,
    ALL_DASHBOARDS,
    DASHBOARD_CUSTOM,
    DASHBOARD_STANDARD,
    QS_DASHBOARDS_STORAGE_KEY,
    QS_DASHBOARDS_STORAGE_VERSION,
    LOVELACE_DATA,
)


class MockResourceStorageCollection:
    """Mock ResourceStorageCollection for testing."""

    def __init__(self, items: list[dict[str, Any]] | None = None):
        self._items = items or []
        self.loaded = False
        self.created_items: list[dict[str, Any]] = []
        self.updated_items: list[tuple[str, dict[str, Any]]] = []
        self.store = MagicMock()
        self.store.key = "lovelace_resources"
        self.store.version = 1

    async def async_load(self):
        """Load the resources."""
        self.loaded = True

    def async_items(self) -> list[dict[str, Any]]:
        """Return all items."""
        return self._items

    async def async_create_item(self, data: dict[str, Any]):
        """Create a new item."""
        self.created_items.append(data)

    async def async_update_item(self, item_id: str, data: dict[str, Any]):
        """Update an existing item."""
        self.updated_items.append((item_id, data))


class MockLovelaceStorage:
    """Mock LovelaceStorage for testing."""

    def __init__(self):
        self.saved_config: dict[str, Any] | None = None
        self.deleted = False

    async def async_save(self, config: dict[str, Any]) -> None:
        """Save config."""
        self.saved_config = config

    async def async_delete(self) -> None:
        """Delete config."""
        self.deleted = True


class MockLovelaceData:
    """Mock LovelaceData for testing."""

    def __init__(self):
        self.dashboards: dict[str | None, Any] = {}
        self.resources = None


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


def create_mock_hass(config_dir: str | None = None) -> MagicMock:
    """Create a mock Home Assistant instance with lovelace data."""
    hass = MagicMock()

    lovelace_data = MockLovelaceData()
    hass.data = {"lovelace": lovelace_data}

    config = MagicMock()
    config_dir = config_dir or tempfile.mkdtemp()
    config.path = MagicMock(
        side_effect=lambda *args: os.path.join(config_dir, *args) if args else config_dir
    )
    hass.config = config

    # Support async_add_executor_job: just runs the function synchronously
    async def _run_in_executor(func, *args):
        return func(*args)

    hass.async_add_executor_job = _run_in_executor

    return hass, lovelace_data


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
    mock_resources.store.version = 2

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

    load_called = False
    original_load = resources.async_load

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
# Tests for _async_copy_and_register_resources
# =============================================================================


@pytest.mark.asyncio
async def test_copy_resources_creates_destination_dir():
    """Test that destination directory is created."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", None
        )

        assert os.path.exists(to_dir)


@pytest.mark.asyncio
async def test_copy_resources_copies_files():
    """Test that files are copied from source to destination."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        test_file = os.path.join(from_dir, "test.js")
        with open(test_file, "w") as f:
            f.write("console.log('test');")

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", None
        )

        dest_file = os.path.join(to_dir, "test.js")
        assert os.path.exists(dest_file)
        with open(dest_file, "r") as f:
            assert f.read() == "console.log('test');"


@pytest.mark.asyncio
async def test_copy_resources_copies_binary_files():
    """Test that binary files are copied correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        test_file = os.path.join(from_dir, "test.bin")
        binary_content = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD])
        with open(test_file, "wb") as f:
            f.write(binary_content)

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", None
        )

        dest_file = os.path.join(to_dir, "test.bin")
        assert os.path.exists(dest_file)
        with open(dest_file, "rb") as f:
            assert f.read() == binary_content


@pytest.mark.asyncio
async def test_copy_resources_recursive_directories():
    """Test that subdirectories are copied recursively."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        subdir = os.path.join(from_dir, "subdir")
        os.makedirs(subdir)

        with open(os.path.join(from_dir, "root.js"), "w") as f:
            f.write("root")
        with open(os.path.join(subdir, "sub.js"), "w") as f:
            f.write("sub")

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", None
        )

        assert os.path.exists(os.path.join(to_dir, "root.js"))
        assert os.path.exists(os.path.join(to_dir, "subdir", "sub.js"))


@pytest.mark.asyncio
async def test_copy_resources_registers_resources():
    """Test that resources are registered with resource handler."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        with open(os.path.join(from_dir, "test1.js"), "w") as f:
            f.write("test1")
        with open(os.path.join(from_dir, "test2.js"), "w") as f:
            f.write("test2")

        resources = MockResourceStorageCollection()

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", resources
        )

        assert len(resources.created_items) == 2


@pytest.mark.asyncio
async def test_copy_resources_registers_with_qs_tag():
    """Test that resources are registered with qs_tag parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        with open(os.path.join(from_dir, "test.js"), "w") as f:
            f.write("test")

        resources = MockResourceStorageCollection()

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "12345", resources
        )

        assert len(resources.created_items) == 1
        assert resources.created_items[0]["url"] == "/local/test/test.js?qs_tag=12345"


@pytest.mark.asyncio
async def test_copy_resources_no_register_when_handler_none():
    """Test that nothing is registered when handler is None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        with open(os.path.join(from_dir, "test.js"), "w") as f:
            f.write("test")

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", None
        )

        assert os.path.exists(os.path.join(to_dir, "test.js"))


@pytest.mark.asyncio
async def test_copy_resources_recursive_namespace():
    """Test that namespace is updated correctly for subdirectories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        subdir = os.path.join(from_dir, "cards")
        os.makedirs(subdir)

        with open(os.path.join(subdir, "card.js"), "w") as f:
            f.write("card")

        resources = MockResourceStorageCollection()

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", resources
        )

        assert len(resources.created_items) == 1
        assert resources.created_items[0]["url"] == "/local/test/cards/card.js?qs_tag=123"


@pytest.mark.asyncio
async def test_copy_resources_empty_directory():
    """Test handling of empty source directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from_dir = os.path.join(tmp_dir, "source")
        to_dir = os.path.join(tmp_dir, "dest")
        os.makedirs(from_dir)

        resources = MockResourceStorageCollection()

        await _async_copy_and_register_resources(
            from_dir, to_dir, "/local/test", "123", resources
        )

        assert os.path.exists(to_dir)
        assert len(resources.created_items) == 0


# =============================================================================
# Tests for _async_register_or_update_dashboard
# =============================================================================


@pytest.mark.asyncio
async def test_register_dashboard_creates_new_panel():
    """Test that a new dashboard panel is created when it doesn't exist."""
    mock_hass, lovelace_data = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._register_panel') as mock_register:
            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                mock_store = AsyncMock()
                mock_store_cls.return_value = mock_store

                await _async_register_or_update_dashboard(
                    mock_hass, DASHBOARD_CUSTOM, {"views": []}
                )

                mock_register.assert_called_once()
                mock_store.async_save.assert_called_once_with({"views": []})
                assert "quiet-solar" in lovelace_data.dashboards


@pytest.mark.asyncio
async def test_register_dashboard_updates_existing():
    """Test that an existing dashboard is updated, not duplicated."""
    mock_hass, lovelace_data = create_mock_hass()

    existing_store = AsyncMock()
    lovelace_data.dashboards["quiet-solar"] = existing_store

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._register_panel') as mock_register:
            await _async_register_or_update_dashboard(
                mock_hass, DASHBOARD_CUSTOM, {"views": [{"title": "Updated"}]}
            )

            # Panel should NOT be re-registered
            mock_register.assert_not_called()
            # Content should be saved
            existing_store.async_save.assert_called_once_with(
                {"views": [{"title": "Updated"}]}
            )


@pytest.mark.asyncio
async def test_register_dashboard_handles_panel_conflict():
    """Test fallback to update=True when panel already exists."""
    mock_hass, lovelace_data = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._register_panel') as mock_register:
            # First call raises ValueError, second (update=True) succeeds
            mock_register.side_effect = [ValueError("exists"), None]

            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                mock_store = AsyncMock()
                mock_store_cls.return_value = mock_store

                await _async_register_or_update_dashboard(
                    mock_hass, DASHBOARD_CUSTOM, {"views": []}
                )

                assert mock_register.call_count == 2
                # Second call should have update=True
                _, kwargs = mock_register.call_args
                assert kwargs.get("update") is True


@pytest.mark.asyncio
async def test_register_dashboard_no_lovelace_data():
    """Test graceful handling when lovelace data is not available."""
    mock_hass, _ = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=None):
        # Should not raise
        await _async_register_or_update_dashboard(
            mock_hass, DASHBOARD_CUSTOM, {"views": []}
        )


# =============================================================================
# Tests for async_restore_dashboards
# =============================================================================


@pytest.mark.asyncio
async def test_restore_dashboards_no_lovelace_data():
    """Test that restore does nothing when lovelace is not ready."""
    mock_hass, _ = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=None):
        # Should not raise
        await async_restore_dashboards_and_update_resources(mock_hass)


@pytest.mark.asyncio
async def test_restore_dashboards_no_tracking_data():
    """Test that restore does nothing when no dashboards were previously generated."""
    mock_hass, lovelace_data = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=None):
            with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                await async_restore_dashboards_and_update_resources(mock_hass)

                assert len(lovelace_data.dashboards) == 0


@pytest.mark.asyncio
async def test_restore_dashboards_registers_panels():
    """Test that restore creates panels from tracking data."""
    mock_hass, lovelace_data = create_mock_hass()
    mock_hass.data["frontend_panels"] = {}  # No panels yet

    tracking_data = {
        "dashboards": [DASHBOARD_CUSTOM["id"], DASHBOARD_STANDARD["id"]],
    }

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=tracking_data):
            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_ll_store:
                mock_ll_store.return_value = MagicMock()

                with patch('custom_components.quiet_solar.ui.dashboard._register_panel') as mock_register:
                    with patch('custom_components.quiet_solar.ui.dashboard.frontend') as mock_frontend:
                        mock_frontend.DATA_PANELS = "frontend_panels"
                        with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                            await async_restore_dashboards_and_update_resources(mock_hass)

                            assert mock_register.call_count == 2
                            assert "quiet-solar" in lovelace_data.dashboards
                            assert "quiet-solar-standard" in lovelace_data.dashboards


@pytest.mark.asyncio
async def test_restore_dashboards_skips_already_registered():
    """Test that restore skips dashboards already in lovelace data."""
    mock_hass, lovelace_data = create_mock_hass()
    lovelace_data.dashboards["quiet-solar"] = MagicMock()  # Already exists

    tracking_data = {
        "dashboards": [DASHBOARD_CUSTOM["id"], DASHBOARD_STANDARD["id"]],
    }

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=tracking_data):
            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_ll_store:
                mock_ll_store.return_value = MagicMock()

                with patch('custom_components.quiet_solar.ui.dashboard._register_panel') as mock_register:
                    with patch('custom_components.quiet_solar.ui.dashboard.frontend') as mock_frontend:
                        mock_frontend.DATA_PANELS = "frontend_panels"
                        with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                            await async_restore_dashboards_and_update_resources(mock_hass)

                            # Only the standard dashboard should be registered (custom already exists)
                            mock_register.assert_called_once()
                            assert "quiet-solar-standard" in lovelace_data.dashboards


@pytest.mark.asyncio
async def test_restore_always_updates_resources():
    """Test that JS resources are always updated on restore, even without tracking data."""
    mock_hass, lovelace_data = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=None):
            with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock) as mock_update:
                await async_restore_dashboards_and_update_resources(mock_hass)

                mock_update.assert_called_once_with(mock_hass)


# =============================================================================
# Tests for async_auto_generate_if_first_install
# =============================================================================


@pytest.mark.asyncio
async def test_auto_generate_first_install():
    """Test that dashboards are generated on first install."""
    mock_hass, lovelace_data = create_mock_hass()
    mock_home = MagicMock()
    mock_home.hass = mock_hass

    with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=None):
        with patch('custom_components.quiet_solar.ui.dashboard.generate_dashboard_yaml', new_callable=AsyncMock) as mock_gen:
            await async_auto_generate_if_first_install(mock_home)

            mock_gen.assert_called_once_with(mock_home)


@pytest.mark.asyncio
async def test_auto_generate_skips_if_already_generated():
    """Test that auto-generation is skipped if dashboards were already generated."""
    mock_hass, lovelace_data = create_mock_hass()
    mock_home = MagicMock()
    mock_home.hass = mock_hass

    tracking_data = {"dashboards": [DASHBOARD_CUSTOM["id"]]}

    with patch('custom_components.quiet_solar.ui.dashboard._async_load_tracking', return_value=tracking_data):
        with patch('custom_components.quiet_solar.ui.dashboard.generate_dashboard_yaml', new_callable=AsyncMock) as mock_gen:
            await async_auto_generate_if_first_install(mock_home)

            mock_gen.assert_not_called()


# =============================================================================
# Tests for async_unregister_dashboards
# =============================================================================


@pytest.mark.asyncio
async def test_unregister_dashboards_removes_panels():
    """Test that unregister removes panels and storage."""
    mock_hass, lovelace_data = create_mock_hass()

    mock_store_1 = MockLovelaceStorage()
    mock_store_2 = MockLovelaceStorage()
    lovelace_data.dashboards["quiet-solar"] = mock_store_1
    lovelace_data.dashboards["quiet-solar-standard"] = mock_store_2

    with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
        with patch('custom_components.quiet_solar.ui.dashboard.frontend') as mock_frontend:
            with patch('custom_components.quiet_solar.ui.dashboard.Store') as mock_tracking_store_cls:
                mock_tracking_store = AsyncMock()
                mock_tracking_store_cls.return_value = mock_tracking_store

                await async_unregister_dashboards(mock_hass)

                # Panels should be removed
                assert mock_frontend.async_remove_panel.call_count == 2

                # Lovelace stores should be deleted
                assert mock_store_1.deleted
                assert mock_store_2.deleted

                # Tracking store should be removed
                mock_tracking_store.async_remove.assert_called_once()

                # Dashboards dict should be empty
                assert "quiet-solar" not in lovelace_data.dashboards
                assert "quiet-solar-standard" not in lovelace_data.dashboards


# =============================================================================
# Tests for generate_dashboard_yaml (integration tests)
# =============================================================================


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_renders_and_saves():
    """Test that generate_dashboard_yaml renders templates and saves to storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        rendered_yaml = "views:\n  - type: sections\n    title: Test"

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            mock_tpl.async_render.return_value = rendered_yaml
            mock_tpl_cls.return_value = mock_tpl

            with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
                with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                    with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock) as mock_save_tracking:
                            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                                mock_store = AsyncMock()
                                mock_store_cls.return_value = mock_store

                                with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                    await generate_dashboard_yaml(mock_home)

                                # Both dashboards should have been saved
                                assert mock_store.async_save.call_count == 2
                                # The saved config should be the parsed YAML
                                saved_config = mock_store.async_save.call_args_list[0][0][0]
                                assert "views" in saved_config
                                assert saved_config["views"][0]["type"] == "sections"

                                # Tracking should have been saved
                                mock_save_tracking.assert_called_once()


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_passes_home_to_template():
    """Test that home object is passed to the template."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        captured_variables = []

        def capture_render(variables):
            captured_variables.append(variables)
            return "views:\n  - title: test"

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            mock_tpl.async_render.side_effect = capture_render
            mock_tpl_cls.return_value = mock_tpl

            with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
                with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                    with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                            with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                                mock_store_cls.return_value = AsyncMock()
                                with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                    await generate_dashboard_yaml(mock_home)

        assert len(captured_variables) == 2
        assert captured_variables[0]["home"] is mock_home
        assert captured_variables[1]["home"] is mock_home


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_handles_template_error():
    """Test that template errors are properly raised."""
    from homeassistant.exceptions import TemplateError

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            mock_tpl.async_render.side_effect = TemplateError("Test template error")
            mock_tpl_cls.return_value = mock_tpl

            with pytest.raises(TemplateError):
                await generate_dashboard_yaml(mock_home)


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_handles_yaml_parse_error():
    """Test that YAML parse errors are properly raised."""
    import yaml

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            # Return invalid YAML that will cause a parse error
            mock_tpl.async_render.return_value = ":\n  invalid: :\n  yaml: {[}"
            mock_tpl_cls.return_value = mock_tpl

            with pytest.raises(yaml.YAMLError):
                await generate_dashboard_yaml(mock_home)


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_skips_non_dict_result():
    """Test that a template producing a non-dict is skipped gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        call_count = 0

        def render_side_effect(variables):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "just a string"  # Not a dict
            return "views:\n  - title: ok"

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            mock_tpl.async_render.side_effect = render_side_effect
            mock_tpl_cls.return_value = mock_tpl

            with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
                with patch('custom_components.quiet_solar.ui.dashboard._async_register_or_update_dashboard', new_callable=AsyncMock) as mock_register:
                    with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                        with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                            with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                                await generate_dashboard_yaml(mock_home)

                    # Only the second template (valid dict) should have been registered
                    assert mock_register.call_count == 1


@pytest.mark.asyncio
async def test_generate_dashboard_yaml_copies_resources():
    """Test that resources are copied during generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = MockHome(mock_hass)

        with patch('custom_components.quiet_solar.ui.dashboard.Template') as mock_tpl_cls:
            mock_tpl = MagicMock()
            mock_tpl.async_render.return_value = "views:\n  - title: test"
            mock_tpl_cls.return_value = mock_tpl

            with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
                with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock) as mock_update_res:
                    with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                            mock_store_cls.return_value = AsyncMock()
                            with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                await generate_dashboard_yaml(mock_home)

                    mock_update_res.assert_called_once_with(mock_hass)


# =============================================================================
# Tests for _async_save_dashboard_tracking
# =============================================================================


@pytest.mark.asyncio
async def test_save_dashboard_tracking():
    """Test that tracking data is saved correctly."""
    mock_hass, _ = create_mock_hass()

    with patch('custom_components.quiet_solar.ui.dashboard.Store') as mock_store_cls:
        mock_store = AsyncMock()
        mock_store_cls.return_value = mock_store

        await _async_save_dashboard_tracking(mock_hass)

        mock_store_cls.assert_called_once_with(
            mock_hass, QS_DASHBOARDS_STORAGE_VERSION, QS_DASHBOARDS_STORAGE_KEY
        )
        mock_store.async_save.assert_called_once()
        saved_data = mock_store.async_save.call_args[0][0]
        assert "dashboards" in saved_data
        assert DASHBOARD_CUSTOM["id"] in saved_data["dashboards"]
        assert DASHBOARD_STANDARD["id"] in saved_data["dashboards"]


# =============================================================================
# Real Template Rendering Tests (No Mocking of templates)
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
        self.car_charge_percent_sensor = None

    def get_attached_virtual_devices(self):
        """Return empty list for virtual devices."""
        return []

    def can_use_charge_percent_constraints(self):
        """For car devices."""
        return True


def create_real_home_for_dashboard(hass, lovelace_data, dashboard_sections=None):
    """Create a more realistic home object for dashboard testing."""
    home = MockHome(hass, dashboard_sections or [("Main", "mdi:home"), ("Cars", "mdi:car")])

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

    devices = [car, climate, pool, battery, person, home_device]

    for device in devices:
        device.home = home

    home.devices = devices
    return home


@pytest.mark.asyncio
async def test_real_template_rendering_produces_valid_lovelace():
    """Test actual template rendering produces a valid Lovelace config dict."""
    import yaml

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass, lovelace_data)

        saved_configs: list[dict[str, Any]] = []

        with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                    with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                            mock_store = AsyncMock()

                            async def capture_save(config):
                                saved_configs.append(config)

                            mock_store.async_save.side_effect = capture_save
                            mock_store_cls.return_value = mock_store

                            with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                await generate_dashboard_yaml(mock_home)

        # Both dashboards should have been saved
        assert len(saved_configs) == 2

        for config in saved_configs:
            assert isinstance(config, dict), "Config should be a dict"
            assert "views" in config, "Config should have 'views' key"
            assert isinstance(config["views"], list), "Views should be a list"
            assert len(config["views"]) > 0, "Should have at least one view"


@pytest.mark.asyncio
async def test_real_template_rendering_contains_devices():
    """Test that rendered dashboard config contains device information."""
    import yaml
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass, lovelace_data)

        saved_configs: list[dict[str, Any]] = []

        with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                    with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                            mock_store = AsyncMock()

                            async def capture_save(config):
                                saved_configs.append(config)

                            mock_store.async_save.side_effect = capture_save
                            mock_store_cls.return_value = mock_store

                            with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                await generate_dashboard_yaml(mock_home)

        # Check the custom dashboard (first one)
        config = saved_configs[0]
        content = json.dumps(config)

        # Check for section names
        assert "Main" in content, "Main section not found"
        assert "Cars" in content, "Cars section not found"

        # Check for entity IDs
        assert "sensor.test_car_soc" in content, "Car SoC entity not found"
        assert "switch.heat_pump_enable" in content, "Heat pump enable entity not found"
        assert "sensor.battery_command" in content, "Battery command entity not found"


@pytest.mark.asyncio
async def test_real_template_rendering_valid_view_structure():
    """Test that rendered views have valid Lovelace structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_hass, lovelace_data = create_mock_hass(tmp_dir)
        mock_home = create_real_home_for_dashboard(mock_hass, lovelace_data)

        saved_configs: list[dict[str, Any]] = []

        with patch('custom_components.quiet_solar.ui.dashboard._get_lovelace_data', return_value=lovelace_data):
            with patch('custom_components.quiet_solar.ui.dashboard._get_resource_handler', return_value=None):
                with patch('custom_components.quiet_solar.ui.dashboard.async_update_resources', new_callable=AsyncMock):
                    with patch('custom_components.quiet_solar.ui.dashboard._async_save_dashboard_tracking', new_callable=AsyncMock):
                        with patch('custom_components.quiet_solar.ui.dashboard.lovelace_dashboard.LovelaceStorage') as mock_store_cls:
                            mock_store = AsyncMock()

                            async def capture_save(config):
                                saved_configs.append(config)

                            mock_store.async_save.side_effect = capture_save
                            mock_store_cls.return_value = mock_store

                            with patch('custom_components.quiet_solar.ui.dashboard._register_panel'):
                                await generate_dashboard_yaml(mock_home)

        config = saved_configs[0]
        first_view = config["views"][0]

        assert "type" in first_view, "View doesn't have type"
        assert first_view["type"] == "sections", "View type is not 'sections'"
        assert "title" in first_view, "View doesn't have title"
        assert "sections" in first_view, "View doesn't have sections"
