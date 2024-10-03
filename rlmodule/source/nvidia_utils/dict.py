# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for utilities for working with dictionaries."""

from collections.abc import Iterable, Mapping
from typing import Any

from .string import callable_to_string, string_to_callable

"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = obj.__dict__

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key
        # check if key is present in the object
        if hasattr(obj, key) or isinstance(obj, dict):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            if isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue
            if isinstance(value, Iterable) and not isinstance(value, str):
                # check length of value to be safe
                if len(obj_mem) != len(value) and obj_mem is not None:
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )
                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True
                    # recursively call if iterable contains dictionaries
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], dict):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    # do not set value to obj, otherwise it overwrites the cfg class with the dict
                    if not set_obj:
                        continue
            elif callable(obj_mem):
                # update function name
                value = string_to_callable(value)
            elif isinstance(value, type(obj_mem)) or value is None:
                pass
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )
            # set value
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")
        