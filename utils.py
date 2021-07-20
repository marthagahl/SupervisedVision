import os
import os.path
import pathlib
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple

import torch


T = TypeVar("T", str, bytes)

def verify_str_arg(value: T, arg: Optional[str] = None, valid_values: Iterable[T] = None, custom_msg: Optional[str] = None,) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = 'Expected type str, but got type {type}.'
        else:
            msg = 'Expected type str for argument {arg}, but got type {type}.'

        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ('Unknown value for argument.'
                    'Valid values are {{{valid_values}}}.')
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))

        raise ValueError(msg)

    return value


