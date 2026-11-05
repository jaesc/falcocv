"""Shared PLC write-verification comparison tests."""

from __future__ import annotations

from falcocv.comms import TagDef
from falcocv.comms.drivers.verify import values_match


def _tag(tag_type: str) -> TagDef:
    return TagDef(device="plc", name="value", address="value", type=tag_type)


def test_float_verification_uses_tolerance():
    assert values_match(_tag("float"), 1.23, 1.2300000190734863)
    assert not values_match(_tag("float"), 1.23, 1.25)


def test_string_verification_ignores_nuls_and_padding():
    assert values_match(_tag("str"), "abc", "abc\x00\x00")
    assert not values_match(_tag("str"), "abc", "ab")


def test_integer_and_bool_verification_are_exact():
    assert values_match(_tag("int"), 7, 7)
    assert not values_match(_tag("int"), 7, 8)
    assert values_match(_tag("bool"), True, True)
    assert not values_match(_tag("bool"), True, False)
