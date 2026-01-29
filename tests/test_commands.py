"""Tests for home_model.commands."""
from __future__ import annotations

from custom_components.quiet_solar.home_model.commands import (
    CMD_CST_AUTO_CONSIGN,
    CMD_CST_AUTO_GREEN,
    CMD_CST_OFF,
    CMD_CST_ON,
    CMD_OFF,
    LoadCommand,
    copy_command,
    copy_command_and_change_type,
    merge_commands,
)


def test_is_like_none() -> None:
    """Test is_like returns False for None."""
    cmd = LoadCommand(command=CMD_CST_ON, power_consign=0.0)
    assert cmd.is_like(None) is False


def test_merge_commands_with_none() -> None:
    """Test merge_commands handles None inputs."""
    cmd = LoadCommand(command=CMD_CST_ON, power_consign=100.0)
    assert merge_commands(None, cmd) == cmd
    assert merge_commands(cmd, None) == cmd


def test_merge_commands_score_and_power() -> None:
    """Test merge picks higher score and max power."""
    cmd_high = LoadCommand(command=CMD_CST_ON, power_consign=100.0)
    cmd_low = LoadCommand(command=CMD_CST_OFF, power_consign=200.0)

    merged = merge_commands(cmd_high, cmd_low)
    assert merged.command == CMD_CST_ON
    assert merged.power_consign == 200.0


def test_copy_command_variants() -> None:
    """Test copy helpers for commands."""
    cmd = LoadCommand(command=CMD_CST_AUTO_CONSIGN, power_consign=150.0)
    assert copy_command(None) is None

    copied = copy_command(cmd)
    assert copied == cmd
    assert copied is not cmd

    updated = copy_command(cmd, power_consign=250.0)
    assert updated.command == cmd.command
    assert updated.power_consign == 250.0

    changed = copy_command_and_change_type(cmd, CMD_CST_AUTO_GREEN)
    assert changed.command == CMD_CST_AUTO_GREEN
    assert changed.power_consign == cmd.power_consign

