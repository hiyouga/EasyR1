#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Android Game Integration Verification Script

This script verifies that all components needed for online Android game training
are properly installed and configured.

Usage:
    python3 scripts/verify_android_game_integration.py
"""

import os
import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")


def check_python_version():
    """Check if Python version is compatible"""
    print_section("Python Environment")

    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} is not supported. Requires Python 3.9+")
        return False


def check_pythonpath():
    """Check if EasyR1 is in PYTHONPATH"""
    print_section("PYTHONPATH Configuration")

    easyr1_root = Path(__file__).parent.parent.absolute()
    print_info(f"EasyR1 root: {easyr1_root}")

    if str(easyr1_root) in sys.path:
        print_success("EasyR1 is in Python module search path")
        return True
    else:
        print_warning("EasyR1 is not in sys.path")
        print_info(f"Add to PYTHONPATH: export PYTHONPATH={easyr1_root}:$PYTHONPATH")
        return False


def check_module_imports():
    """Check if all required modules can be imported"""
    print_section("Module Import Checks")

    modules_to_check = [
        ("verl.trainer.main", "Core trainer module"),
        ("verl.workers.fsdp_workers", "FSDP worker module"),
        ("verl.trainer.config", "Configuration module"),
        ("verl.trainer.data_loader", "Data loader module"),
        ("number_game_agent.workers.android_game_rollout", "AndroidGameRollout"),
        ("number_game_agent.data.online_game_dataloader", "OnlineGameDataLoader"),
        ("number_game_agent.reward_function.online_game_reward", "Reward function"),
        ("number_game_agent.workers.game_state_manager", "Game state manager"),
    ]

    all_success = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
        except ImportError as e:
            print_error(f"{description}: {module_name} - {str(e)}")
            all_success = False

    return all_success


def check_adb_devices():
    """Check if ADB is installed and devices are connected"""
    print_section("Android Device Configuration")

    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")

        # First line is "List of devices attached"
        if len(lines) <= 1:
            print_warning("No Android devices connected")
            print_info("Connect an Android device or start an emulator")
            return False

        devices = []
        for line in lines[1:]:
            if line.strip() and "\tdevice" in line:
                device_id = line.split("\t")[0]
                devices.append(device_id)

        if devices:
            print_success(f"Found {len(devices)} Android device(s):")
            for device_id in devices:
                print_info(f"  - {device_id}")
            return True
        else:
            print_warning("ADB devices found but not in 'device' state")
            print_info("Check device authorization (accept on device screen)")
            return False

    except FileNotFoundError:
        print_error("ADB command not found. Please install Android SDK Platform Tools")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"ADB command failed: {e}")
        return False


def check_config_file():
    """Check if example config file exists"""
    print_section("Configuration Files")

    config_path = Path(__file__).parent.parent / "examples" / "config_online_android_game.yaml"

    if config_path.exists():
        print_success(f"Example config found: {config_path}")

        # Read and validate key fields
        with open(config_path, "r") as f:
            content = f.read()

        required_fields = ["online_mode: true", 'name: android_game', "device_pool:"]

        all_present = True
        for field in required_fields:
            if field in content:
                print_success(f"  Config contains: {field}")
            else:
                print_error(f"  Config missing: {field}")
                all_present = False

        return all_present
    else:
        print_error(f"Example config not found: {config_path}")
        return False


def check_gpu_availability():
    """Check if CUDA GPUs are available"""
    print_section("GPU Configuration")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print_success("NVIDIA GPU(s) detected")

        # Extract GPU count
        gpu_lines = [line for line in result.stdout.split("\n") if "MiB" in line and "|" in line]
        print_info(f"  Found {len(gpu_lines)} GPU(s)")
        return True

    except FileNotFoundError:
        print_warning("nvidia-smi not found. GPU may not be available")
        return False
    except subprocess.CalledProcessError:
        print_warning("nvidia-smi failed. GPU may not be configured correctly")
        return False


def check_framework_modifications():
    """Verify framework files have correct modifications"""
    print_section("Framework Modifications")

    easyr1_root = Path(__file__).parent.parent

    files_to_check = [
        ("verl/trainer/config.py", ["online_mode: bool = False", "device_pool:"]),
        ("verl/trainer/data_loader.py", ["if hasattr(config, 'online_mode')", "create_online_game_dataloader"]),
        ("verl/workers/rollout/config.py", ["device_pool:", "game_config:"]),
        ("verl/workers/fsdp_workers.py", ['if self.config.rollout.name == "android_game"', "AndroidGameRollout"]),
    ]

    all_present = True
    for file_path, required_strings in files_to_check:
        full_path = easyr1_root / file_path
        if not full_path.exists():
            print_error(f"{file_path}: File not found")
            all_present = False
            continue

        with open(full_path, "r") as f:
            content = f.read()

        file_ok = True
        for required_str in required_strings:
            if required_str not in content:
                print_error(f"{file_path}: Missing '{required_str}'")
                file_ok = False
                all_present = False

        if file_ok:
            print_success(f"{file_path}: All modifications present")

    return all_present


def check_dependencies():
    """Check if required Python packages are installed"""
    print_section("Python Dependencies")

    required_packages = [
        "torch",
        "transformers",
        "vllm",
        "ray",
        "tensordict",
        "omegaconf",
        "jinja2",
        "Pillow",
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} - Not installed")
            all_installed = False

    return all_installed


def main():
    """Run all verification checks"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("Android Game Integration Verification".center(60))
    print("=" * 60)
    print(f"{Colors.RESET}")

    checks = [
        ("Python Version", check_python_version),
        ("PYTHONPATH", check_pythonpath),
        ("Dependencies", check_dependencies),
        ("Module Imports", check_module_imports),
        ("Framework Modifications", check_framework_modifications),
        ("Configuration File", check_config_file),
        ("Android Devices", check_adb_devices),
        ("GPU Availability", check_gpu_availability),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Check '{name}' failed with exception: {e}")
            results[name] = False

    # Print summary
    print_section("Verification Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:6}{Colors.RESET} {name}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} checks passed{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All checks passed! Ready to start training.{Colors.RESET}")
        print(f"\n{Colors.BLUE}Next steps:{Colors.RESET}")
        print("  1. Review config: examples/config_online_android_game.yaml")
        print("  2. Start training: python3 -m verl.trainer.main config=examples/config_online_android_game.yaml")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Some checks failed. Please fix the issues above.{Colors.RESET}")
        print(f"\n{Colors.BLUE}Documentation:{Colors.RESET}")
        print("  - Quick Start: examples/QUICKSTART_ONLINE_GAME.md")
        print("  - Integration Guide: number_game_agent/INTEGRATION_SUMMARY.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
