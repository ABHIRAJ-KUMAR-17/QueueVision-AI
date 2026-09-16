"""
conftest.py – pytest configuration for QueueVision AI.

Ensures the project root is on sys.path so that `src.*` imports
work correctly when tests are run from any working directory.
"""
import sys
import os

# Insert project root (one level up from this tests/ directory) at position 0
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
