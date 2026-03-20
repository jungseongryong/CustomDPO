# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Optional

import trl

try:
    _ORPOConfigBase = trl.ORPOConfig
except AttributeError:
    from trl.experimental.orpo import ORPOConfig as _ORPOConfigBase


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )
    teacher_prompt_instruction: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional instruction inserted into teacher/ref prompts for custom DPO training.",
        },
    )
    teacher_chosen_prompt_instruction: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional instruction inserted only into the teacher/ref prompt for the chosen branch.",
        },
    )
    teacher_rejected_prompt_instruction: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional instruction inserted only into the teacher/ref prompt for the rejected branch.",
        },
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional separate teacher/reference model for TeacherPromptAlignedDPOTrainer. "
                "This model must share the same token-id space as the student when used with token-level KL."
            ),
        },
    )
    teacher_tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional tokenizer path for the teacher/reference model. If unset, the teacher model path is used "
                "for tokenizer compatibility checks."
            ),
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    rkl_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Rejected-side reverse-KL mixing weight for `distillm2_token`. If unset, the trainer uses 0.5 so "
                "the chosen forward KL and rejected reverse KL are weighted equally."
            )
        },
    )
    trust_region_alpha: float = field(
        default=0.0,
        metadata={
            "help": (
                "Mixing weight for the trust-region teacher used by `distillm2_token`. "
                "0 keeps the fixed reference teacher, while values in (0, 1] geometrically mix it "
                "with the current model's teacher-prompt distribution under stop-gradient."
            )
        },
    )


@dataclass
class ORPOConfig(_ORPOConfigBase):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
