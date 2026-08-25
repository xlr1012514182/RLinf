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

import numpy as np

from rlinf.projects.fibocom_vla.rl.gae import compute_gae, normalize_masked
from rlinf.projects.fibocom_vla.rl.rewards import truncate_after_second_failure


def test_second_failure_is_terminal_and_tail_is_zero_masked() -> None:
    result = truncate_after_second_failure(
        rewards=np.array([0, 0, 0, 1, 0], dtype=np.float32),
        failure_events=np.array([False, True, False, True, False]),
        terminal_penalty=-1.0,
    )
    np.testing.assert_array_equal(result.rewards, [0, 0, 0, -1, 0])
    np.testing.assert_array_equal(result.valid_mask, [True, True, True, True, False])
    np.testing.assert_array_equal(result.terminals, [False, False, False, True, False])
    np.testing.assert_array_equal(result.second_failure_indices, [3])


def test_terminal_failure_produces_negative_advantage() -> None:
    rewards = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    terminals = np.array([False, False, True, False])
    valid = np.array([True, True, True, False])
    values = np.zeros(5, dtype=np.float32)
    advantages, returns = compute_gae(
        rewards, values, terminals, valid, gamma=0.99, gae_lambda=0.95
    )
    assert advantages[2] == -1.0
    assert advantages[1] < 0
    assert advantages[0] < 0
    assert advantages[3] == 0
    np.testing.assert_allclose(advantages, returns)


def test_masked_normalization_keeps_padding_zero() -> None:
    values = np.array([[1.0, 2.0, 999.0]], dtype=np.float32)
    mask = np.array([[True, True, False]])
    normalized = normalize_masked(values, mask)
    np.testing.assert_allclose(normalized[0, :2], [-1.0, 1.0], atol=1e-6)
    assert normalized[0, 2] == 0.0

