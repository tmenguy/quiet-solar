# Testing Guide for Quiet Solar

## Prerequisites

### Install Test Dependencies

You have two options:

#### Option 1: Install from requirements file (Recommended)
```bash
pip install -r requirements_test.txt
```

#### Option 2: Install manually
```bash
pip install pytest pytest-asyncio pytest-cov pytest-homeassistant-custom-component
```

### Install Project in Development Mode

From the project root:
```bash
pip install -e .
```

Or if you're working within a Home Assistant development environment, ensure the project is in your Python path.

## Fixing the Import Error

If you see `ModuleNotFoundError: No module named 'custom_components'`, you have several solutions:

### Solution 1: Use the Test Runner Script (Easiest)
```bash
./run_tests.sh
```

The script automatically sets up PYTHONPATH for you.

### Solution 2: Set PYTHONPATH manually
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Solution 3: Use pytest.ini (Already configured)
The `pytest.ini` file in the project root already configures the Python path. Just make sure you're running pytest from the project root:
```bash
cd /Users/tmenguy/Developer/homeassistant/quiet-solar
pytest tests/
```

### Solution 4: Install in Development Mode
```bash
pip install -e .
```

This makes `custom_components` importable system-wide.

## Running Tests

### Quick Start

Run all tests:
```bash
./run_tests.sh
```

Run only new integration tests:
```bash
./run_tests.sh new
```

Run only existing tests:
```bash
./run_tests.sh existing
```

Run with coverage:
```bash
./run_tests.sh coverage
```

Quick smoke test:
```bash
./run_tests.sh quick
```

### Using pytest directly

After setting PYTHONPATH, you can use pytest commands:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_integration_config_flow.py -v

# Specific test function
pytest tests/test_entity.py::test_create_device_from_type_unknown_type -v

# With coverage
pytest tests/ --cov=custom_components.quiet_solar --cov-report=html

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Run only async tests
pytest tests/ -m asyncio

# Run tests matching pattern
pytest tests/ -k "config_flow"
```

## Test Organization

### New Integration Tests
- `test_integration_config_flow.py` - Config flow and options
- `test_integration_init.py` - Setup, teardown, reload
- `test_data_handler.py` - Device management
- `test_entity.py` - Base entity classes
- `test_platform_sensor.py` - Sensor platform
- `test_platform_button.py` - Button platform
- `test_platform_switch.py` - Switch platform
- `test_config_flow_helpers.py` - Config flow utilities

### Existing Tests
- `test_chargers*.py` - Charger logic (3 files)
- `test_solver*.py` - Optimization algorithms (2 files)
- `test_cars.py` - Car management
- `test_forecasts.py` - Forecast calculations
- `test_devices_utils.py` - Device utilities

## Troubleshooting

### ModuleNotFoundError: No module named 'homeassistant'

This means Home Assistant core isn't installed. Options:

1. **Use pytest-homeassistant-custom-component** (recommended):
   ```bash
   pip install pytest-homeassistant-custom-component
   ```
   This provides mock fixtures for testing without full HA installation.

2. **Install Home Assistant core**:
   ```bash
   pip install homeassistant
   ```

3. **Work within HA dev environment**: If you're developing within a Home Assistant installation, activate that environment first.

### Tests pass individually but fail when run together

This suggests state leakage between tests. Check:
- Are fixtures properly isolated?
- Are there any global variables being modified?
- Do tests need `autouse` fixtures for cleanup?

### Async test warnings

Make sure tests are marked with `@pytest.mark.asyncio`:
```python
@pytest.mark.asyncio
async def test_my_async_function():
    ...
```

The `pytest.ini` is configured with `asyncio_mode = auto` so this should work automatically.

### Coverage not showing all files

Make sure you're running from the project root and the source directory structure is correct:
```bash
cd /Users/tmenguy/Developer/homeassistant/quiet-solar
pytest tests/ --cov=custom_components.quiet_solar
```

## Writing New Tests

### Use Existing Fixtures

See `conftest.py` for available fixtures:
- `fake_hass` - Mock Home Assistant instance
- `mock_config_entry` - Generic config entry
- `mock_home_config_entry` - Home-specific config entry
- `mock_charger_config_entry` - Charger-specific config entry
- `mock_car_config_entry` - Car-specific config entry
- `current_time` - Consistent test timestamp
- `mock_data_handler` - Mock data handler

### Example Test

```python
import pytest
from custom_components.quiet_solar.const import DOMAIN

@pytest.mark.asyncio
async def test_my_feature(fake_hass, mock_config_entry):
    """Test description."""
    # Arrange
    fake_hass.data[DOMAIN]["test"] = "value"
    
    # Act
    result = await my_function(fake_hass)
    
    # Assert
    assert result is True
```

### Test Naming Conventions

- File: `test_<module_name>.py`
- Function: `test_<what_it_tests>`
- Use descriptive names: `test_sensor_becomes_unavailable_when_device_disabled`

## Continuous Integration

### GitHub Actions

If setting up CI, use this workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements_test.txt
          pip install -e .
      - name: Run tests
        run: pytest tests/ --cov=custom_components.quiet_solar
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Coverage Goals

Current coverage targets:
- Integration components (config_flow, __init__, data_handler): **70%+**
- Entity components: **75%+**
- Platform components (sensor, button, switch): **70%+**
- Domain logic (ha_model, home_model): **60%+** (target for future)

## Getting Help

1. Check this guide first
2. Review existing tests for patterns
3. Check pytest documentation: https://docs.pytest.org/
4. Check Home Assistant testing docs: https://developers.home-assistant.io/docs/development_testing

## Contributing

When adding new tests:
1. Follow existing patterns in `conftest.py`
2. Use descriptive test names
3. Test both success and failure paths
4. Add docstrings to test functions
5. Update coverage goals if adding new modules
6. Run `./run_tests.sh coverage` before committing
