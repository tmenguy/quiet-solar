#!/usr/bin/env python3
"""Test runner for quiet_solar with proper Python path setup."""

import sys
import os
from pathlib import Path
import subprocess

# Check if running in venv313
if 'venv313' not in sys.executable:
    print("⚠️  Not running in venv313, attempting to activate...")
    project_root = Path(__file__).parent
    venv_python = project_root / 'venv313' / 'bin' / 'python'
    
    if venv_python.exists():
        print(f"✓ Found venv313, re-running with: {venv_python}")
        # Re-run this script with the venv python
        result = subprocess.run([str(venv_python), __file__] + sys.argv[1:])
        sys.exit(result.returncode)
    else:
        print(f"❌ Could not find venv313 at {venv_python}")
        print("Please activate venv313 manually:")
        print("  source venv313/bin/activate")
        sys.exit(1)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now try to import and run pytest
try:
    import pytest
except ImportError:
    print("❌ ERROR: pytest is not installed")
    print("\nPlease install test requirements:")
    print("  pip install -r requirements_test.txt")
    print("\nOr install pytest directly:")
    print("  pip install pytest pytest-asyncio pytest-cov")
    sys.exit(1)

# Check if custom_components can be imported
try:
    from custom_components.quiet_solar import const
    print(f"✓ Successfully imported custom_components.quiet_solar")
    print(f"  Domain: {const.DOMAIN}\n")
except ImportError as e:
    print(f"⚠️  Warning: Could not import custom_components: {e}")
    print("  Tests may fail if dependencies are missing.\n")

def main():
    """Run tests with various options."""
    args = sys.argv[1:]
    
    if not args or args[0] in ['-h', '--help', 'help']:
        print("Quiet Solar Test Runner")
        print("\nUsage:")
        print("  python run_tests.py [options]")
        print("\nOptions:")
        print("  new         - Run only new integration tests")
        print("  existing    - Run only existing tests")
        print("  coverage    - Run all tests with coverage report")
        print("  quick       - Run quick smoke tests")
        print("  <file>      - Run specific test file")
        print("  -v          - Verbose output")
        print("  -x          - Stop on first failure")
        print("  -k <pattern>- Run tests matching pattern")
        print("\nExamples:")
        print("  python run_tests.py new -v")
        print("  python run_tests.py coverage")
        print("  python run_tests.py tests/test_entity.py")
        return 0
    
    # Build pytest arguments
    pytest_args = []
    
    if args[0] == 'new':
        print("🧪 Running new integration tests...\n")
        pytest_args = [
            'tests/test_integration_config_flow.py',
            'tests/test_integration_init.py',
            'tests/test_data_handler.py',
            'tests/test_entity.py',
            'tests/test_platform_sensor.py',
            'tests/test_platform_button.py',
            'tests/test_platform_switch.py',
            'tests/test_config_flow_helpers.py',
            '-v'
        ]
        pytest_args.extend(args[1:])
        
    elif args[0] == 'existing':
        print("🧪 Running existing tests...\n")
        pytest_args = [
            'tests/test_chargers.py',
            'tests/test_chargers_advanced.py',
            'tests/test_chargers_comprehensive.py',
            'tests/test_solver.py',
            'tests/test_solver_2.py',
            'tests/test_cars.py',
            'tests/test_forecasts.py',
            'tests/test_devices_utils.py',
            '-v'
        ]
        pytest_args.extend(args[1:])
        
    elif args[0] == 'coverage':
        print("🧪 Running all tests with coverage...\n")
        pytest_args = [
            'tests/',
            '--cov=custom_components.quiet_solar',
            '--cov-report=term-missing',
            '--cov-report=html',
            '-v'
        ]
        pytest_args.extend(args[1:])
        
    elif args[0] == 'quick':
        print("🧪 Running quick smoke tests...\n")
        pytest_args = [
            'tests/test_config_flow_helpers.py',
            'tests/test_entity.py',
            '-v'
        ]
        pytest_args.extend(args[1:])
        
    else:
        # Pass through all arguments to pytest
        pytest_args = args
    
    # Run pytest
    print(f"Running: pytest {' '.join(pytest_args)}\n")
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return exit_code

if __name__ == '__main__':
    sys.exit(main())
