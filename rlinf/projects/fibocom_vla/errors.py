# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Domain-specific exceptions for the Fibocom VLA stack."""


class FibocomVLAError(RuntimeError):
    """Base class for recoverable stack errors."""


class ConfigurationError(FibocomVLAError):
    """Raised when a configuration is internally inconsistent."""


class HardwareNotReadyError(FibocomVLAError):
    """Raised when a command is attempted before hardware is ready."""


class SafetyViolationError(FibocomVLAError):
    """Raised when an action fails a configured safety gate."""


class ShapeMismatchError(FibocomVLAError):
    """Raised when an observation or action violates its declared schema."""


class OptionalDependencyError(FibocomVLAError, ImportError):
    """Raised when an optional hardware or acceleration dependency is absent."""
