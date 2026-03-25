#!/usr/bin/env python
"""
Wrapper script to run admm_input_group method using the original Feature-Selection-Benchmark framework.

This script interfaces with the original benchmark by:
1. Using the original data loading and evaluation framework
2. Calling your custom admm_input_group implementation
3. Producing results compatible with the original benchmark format
"""

import sys
import os

# Add custom_admm to path to import your implementation
CUSTOM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUSTOM_ROOT)

# Import your custom implementation
from src.admm_input_group_wrapper import run_admm_input_group


def main():
    """Main entry point for running admm_input_group."""
    # This is a template - you can adapt based on your specific needs
    # The actual integration will depend on how you want to call it
    print("Custom ADMM Input Group wrapper ready")
    print("Use this directory to run your method alongside the original benchmark")


if __name__ == "__main__":
    main()
