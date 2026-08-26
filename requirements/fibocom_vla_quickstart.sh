#!/usr/bin/env bash

# Copyright 2025 The RLinf Authors.
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

set -euo pipefail

readonly UV_VERSION="0.12.6"
readonly PYTHON_VERSION="3.11.14"
readonly DEFAULT_PYTHON_INSTALL_MIRROR="https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
TOOLS_DIR="${REPOSITORY_ROOT}/.fibocom-tools/uv-${UV_VERSION}"
UV_BIN="${TOOLS_DIR}/bin/uv"
PYTHON_INSTALL_DIR="${TOOLS_DIR}/python"
MANAGED_PYTHON_BIN="${PYTHON_INSTALL_DIR}/cpython-${PYTHON_VERSION}-linux-x86_64-gnu/bin/python3.11"
PYTHON_INSTALL_MIRROR="${FIBOCOM_PYTHON_INSTALL_MIRROR:-${DEFAULT_PYTHON_INSTALL_MIRROR}}"
VENV_DIR="${FIBOCOM_VENV_DIR:-${REPOSITORY_ROOT}/.venv-fibocom}"
PYTHON_BIN="${VENV_DIR}/bin/python"
REQUIREMENTS_FILE="${SCRIPT_DIR}/fibocom_vla_quickstart.txt"

if [[ "$(pwd -P)" != "${REPOSITORY_ROOT}" ]]; then
    echo "Run this script from the repository root: ${REPOSITORY_ROOT}" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to bootstrap the pinned uv executable." >&2
    exit 1
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "python3-pip is required to bootstrap the pinned uv executable." >&2
    exit 1
fi

if [[ ! -x "${UV_BIN}" ]]; then
    mkdir -p "${TOOLS_DIR}"
    python3 -m pip install \
        --disable-pip-version-check \
        --index-url "https://pypi.org/simple" \
        --no-warn-script-location \
        --upgrade \
        --target "${TOOLS_DIR}" \
        "uv==${UV_VERSION}"
fi

observed_uv_version="$("${UV_BIN}" --version | awk '{print $2}')"
if [[ "${observed_uv_version}" != "${UV_VERSION}" ]]; then
    echo "Expected uv ${UV_VERSION}, found ${observed_uv_version}." >&2
    exit 1
fi

if [[ -e "${VENV_DIR}" && ! -x "${PYTHON_BIN}" ]]; then
    echo "${VENV_DIR} exists but is not a valid Linux virtual environment." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    if [[ ! -x "${MANAGED_PYTHON_BIN}" ]]; then
        "${UV_BIN}" python install "${PYTHON_VERSION}" \
            --cache-dir "${TOOLS_DIR}/cache" \
            --install-dir "${PYTHON_INSTALL_DIR}" \
            --mirror "${PYTHON_INSTALL_MIRROR}" \
            --no-bin
    fi
    "${UV_BIN}" venv "${VENV_DIR}" --python "${MANAGED_PYTHON_BIN}"
fi

observed_python_version="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
if [[ "${observed_python_version}" != "${PYTHON_VERSION}" ]]; then
    echo "Expected Python ${PYTHON_VERSION}, found ${observed_python_version}." >&2
    echo "Set FIBOCOM_VENV_DIR to a new path and rerun the script." >&2
    exit 1
fi

"${UV_BIN}" pip sync \
    --default-index "https://pypi.org/simple" \
    --python "${PYTHON_BIN}" \
    "${REQUIREMENTS_FILE}"

echo "Fibocom CPU quick-start environment is ready."
echo "Interpreter: ${PYTHON_BIN}"
