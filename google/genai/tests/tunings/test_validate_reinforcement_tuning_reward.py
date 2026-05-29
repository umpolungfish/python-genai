# Copyright 2025 Google LLC
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
#


"""Tests for tunings.validate_reinforcement_tuning_reward()."""

from ... import types as genai_types
from .. import pytest_helper

_SAMPLE_RESPONSE = genai_types.Content(
    role="model",
    parts=[genai_types.Part(text="Paris")],
)

_EXAMPLE = genai_types.ReinforcementTuningExample(
    contents=[
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="What is the capital of France?")],
        )
    ],
    references={"answer": "Paris"},
)


test_table: list[pytest_helper.TestTableItem] = [
    pytest_helper.TestTableItem(
        name="test_validate_single_reward_autorater",
        parameters=genai_types.ValidateReinforcementTuningRewardParameters(
            parent="projects/801452371447/locations/us-central1",
            sample_response=_SAMPLE_RESPONSE,
            example=_EXAMPLE,
            single_reward_config=genai_types.SingleReinforcementTuningRewardConfig(
                reward_name="autorater_reward",
                autorater_scorer=genai_types.ReinforcementTuningAutoraterScorer(
                    autorater_config=genai_types.AutoraterConfig(
                        autorater_model="test-model"
                    )
                ),
            ),
        ),
        exception_if_mldev=(
            "only supported in Gemini Enterprise Agent Platform mode"
        ),
    ),
    pytest_helper.TestTableItem(
        name="test_validate_single_reward_string_match",
        parameters=genai_types.ValidateReinforcementTuningRewardParameters(
            parent="projects/801452371447/locations/us-central1",
            sample_response=_SAMPLE_RESPONSE,
            example=_EXAMPLE,
            single_reward_config=genai_types.SingleReinforcementTuningRewardConfig(
                reward_name="string_match_reward",
                string_match_reward_scorer=genai_types.ReinforcementTuningStringMatchRewardScorer(
                    correct_answer_reward=1.0,
                    wrong_answer_reward=-1.0,
                    string_match_expression=genai_types.ReinforcementTuningStringMatchRewardScorerStringMatchExpression(
                        match_operation="EXACT_MATCH",
                        expression="{{references.answer}}",
                    ),
                ),
            ),
        ),
        exception_if_mldev=(
            "only supported in Gemini Enterprise Agent Platform mode"
        ),
    ),
    pytest_helper.TestTableItem(
        name="test_validate_composite_reward",
        parameters=genai_types.ValidateReinforcementTuningRewardParameters(
            parent="projects/801452371447/locations/us-central1",
            sample_response=_SAMPLE_RESPONSE,
            example=_EXAMPLE,
            composite_reward_config=genai_types.CompositeReinforcementTuningRewardConfig(
                weighted_reward_configs=[
                    genai_types.CompositeReinforcementTuningRewardConfigWeightedRewardConfig(
                        weight=1.0,
                        reward_config=genai_types.SingleReinforcementTuningRewardConfig(
                            reward_name="autorater_reward",
                            autorater_scorer=genai_types.ReinforcementTuningAutoraterScorer(
                                autorater_config=genai_types.AutoraterConfig(
                                    autorater_model="test-model"
                                )
                            ),
                        ),
                    ),
                ],
            ),
        ),
        exception_if_mldev=(
            "only supported in Gemini Enterprise Agent Platform mode"
        ),
    ),
]

pytestmark = pytest_helper.setup(
    file=__file__,
    globals_for_file=globals(),
    test_method="tunings.validate_reinforcement_tuning_reward",
    test_table=test_table,
)

pytest_plugins = ("pytest_asyncio",)
