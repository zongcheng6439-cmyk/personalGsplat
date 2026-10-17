# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import Strategy
from .anchor_mcmc import AnchorMCMCStrategy
from .anchor_io import read_anchor_capacity_ply, read_dense_anchor_init_ply
from .default import DefaultStrategy
from .mcmc import MCMCStrategy
from .texture_default import TextureDefaultStrategy
from .texture_mcmc import TextureMCMCStrategy

__all__ = [
    "AnchorMCMCStrategy",
    "DefaultStrategy",
    "MCMCStrategy",
    "Strategy",
    "TextureDefaultStrategy",
    "TextureMCMCStrategy",
    "read_anchor_capacity_ply",
    "read_dense_anchor_init_ply",
]
