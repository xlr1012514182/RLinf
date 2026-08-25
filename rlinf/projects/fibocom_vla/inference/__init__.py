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

"""Composable CUDA Graph, speculative, and RTC inference primitives."""

from .openpi_rtc import (
    OPENPI_MODEL_ACTIONS_KEY,
    OPENPI_RTC_DIAGNOSTICS_KEY,
    OPENPI_RTC_PREFIX_FEATURE_KEY,
    OpenPINativeRTCAdapter,
    OpenPIRTCContext,
    masked_mean_prefix_feature,
    predict_action_batch_with_openpi_features,
)
from .rtc import RTCConditioning, build_rtc_conditioning
from .speculative import SpeculativeChunkPolicy, VerificationResult

__all__ = [
    "OPENPI_MODEL_ACTIONS_KEY",
    "OPENPI_RTC_DIAGNOSTICS_KEY",
    "OPENPI_RTC_PREFIX_FEATURE_KEY",
    "OpenPINativeRTCAdapter",
    "OpenPIRTCContext",
    "RTCConditioning",
    "SpeculativeChunkPolicy",
    "VerificationResult",
    "build_rtc_conditioning",
    "masked_mean_prefix_feature",
    "predict_action_batch_with_openpi_features",
]
