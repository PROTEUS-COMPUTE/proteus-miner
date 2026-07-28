# The MIT License (MIT)
# Copyright © 2024 PROTEUS Compute

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""PROTEUS golden set.

The real set lives only on the router machine. It is loaded at runtime from the
JSON file pointed at by GOLDEN_SET_PATH, and is never committed to this
repository nor included in the published image.

The _EXAMPLE_ITEMS below are placeholders so the code runs in development.

JSON format (list of [prompt, reference] pairs):
    [
        ["prompt text...", "reference answer..."],
        ...
    ]
"""

from __future__ import annotations

import json
import os
import random
import typing


class GoldenSet:
    def __init__(self, items: typing.Optional[typing.List[typing.Tuple[str, str]]] = None):
        if items is not None:
            self.items = items
        else:
            path = os.getenv("GOLDEN_SET_PATH")
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.items = [(str(p), str(r)) for p, r in raw]
            else:
                self.items = list(_EXAMPLE_ITEMS)

    def sample(self) -> typing.Tuple[str, str]:
        return random.choice(self.items)

    def add(self, prompt: str, expected: str) -> None:
        self.items.append((prompt, expected))


# Placeholder tasks for development only. NOT the real golden set.
# The real set is loaded from GOLDEN_SET_PATH on the router machine.
_EXAMPLE_ITEMS: typing.List[typing.Tuple[str, str]] = [
    (
        "Explain what a binary search is in one sentence.",
        "Binary search is an algorithm that finds a target in a sorted array by repeatedly halving the search range.",
    ),
    (
        "What does CPU stand for?",
        "CPU stands for Central Processing Unit.",
    ),
    (
        "Write a one-line Python function that returns the factorial of n.",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
    ),
]
