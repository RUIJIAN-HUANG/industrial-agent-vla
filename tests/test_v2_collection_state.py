import json
from pathlib import Path

import pytest

from simulation.v2_collection_state import (
    CollectionStateError,
    ControlToken,
    EpisodeOutcome,
    P01ToS11CollectionStateMachine,
    V2CollectionContract,
    V2FailureCode,
    V2ManualCollectionStateMachine,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "simulation/configs/single_bin_scene_v2.json"

EXPECTED_ORDER = (
    "P01",
    "P02",
    "P03",
    "P04",
    "N01",
    "N02",
    "W01",
    "W02",
)

EXPECTED_SLOTS = {
    "P01": "S11",
    "P02": "S21",
    "P03": "S12",
    "P04": "S22",
    "N01": "S13",
    "N02": "S23",
    "W01": "S14",
    "W02": "S24",
}


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _machine() -> V2ManualCollectionStateMachine:
    contract = V2CollectionContract.from_config(_config())
    return V2ManualCollectionStateMachine(contract)


def _p01_machine() -> P01ToS11CollectionStateMachine:
    contract = V2CollectionContract.from_config(_config())
    return P01ToS11CollectionStateMachine(contract)


def _place_all(machine: V2ManualCollectionStateMachine) -> None:
    for part_id in EXPECTED_ORDER:
        machine.record_part_placement(
            part_id=part_id,
            slot_id=EXPECTED_SLOTS[part_id],
            stable=True,
        )


def _enter_b_only(machine: V2ManualCollectionStateMachine) -> None:
    _place_all(machine)
    machine.enter_handoff_verify(
        bin_at_handoff_center=True,
        bin_stable=True,
        arm_a_gripper_open=True,
        arm_a_clear=True,
    )
    machine.activate_b_only()


def test_contract_reads_formal_order_from_collection() -> None:
    contract = V2CollectionContract.from_config(_config())

    assert contract.scene_id == "single_bin_manual_industrial_v2"
    assert contract.formal_part_order == EXPECTED_ORDER
    assert dict(contract.part_to_slot) == EXPECTED_SLOTS
    assert contract.token_sequence == (
        ControlToken.A_ONLY,
        ControlToken.HANDOFF_VERIFY,
        ControlToken.B_ONLY,
        ControlToken.NONE,
    )


def test_initial_state_allows_only_arm_a() -> None:
    machine = _machine()

    assert machine.token is ControlToken.A_ONLY
    assert machine.next_part_id == "P01"
    machine.require_arm_action("Arm_A")

    with pytest.raises(CollectionStateError) as caught:
        machine.require_arm_action("Arm_B")

    assert caught.value.code is V2FailureCode.INACTIVE_ARM_ACTION
    assert machine.outcome is EpisodeOutcome.FAILED


def test_part_order_violation_fails_closed() -> None:
    machine = _machine()

    with pytest.raises(CollectionStateError) as caught:
        machine.record_part_placement(
            part_id="P02",
            slot_id="S21",
            stable=True,
        )

    assert caught.value.code is V2FailureCode.PART_ORDER_VIOLATION
    assert machine.token is ControlToken.NONE


def test_wrong_slot_fails_closed() -> None:
    machine = _machine()

    with pytest.raises(CollectionStateError) as caught:
        machine.record_part_placement(
            part_id="P01",
            slot_id="S12",
            stable=True,
        )

    assert caught.value.code is V2FailureCode.SLOT_MISMATCH
    assert machine.outcome is EpisodeOutcome.FAILED


def test_unstable_placement_is_not_accepted() -> None:
    machine = _machine()

    with pytest.raises(CollectionStateError) as caught:
        machine.record_part_placement(
            part_id="P01",
            slot_id="S11",
            stable=False,
        )

    assert caught.value.code is V2FailureCode.PART_UNSTABLE
    assert machine.placed_parts == ()


def test_handoff_requires_all_safety_preconditions() -> None:
    machine = _machine()
    _place_all(machine)

    with pytest.raises(CollectionStateError) as caught:
        machine.enter_handoff_verify(
            bin_at_handoff_center=True,
            bin_stable=True,
            arm_a_gripper_open=True,
            arm_a_clear=False,
        )

    assert caught.value.code is V2FailureCode.HANDOFF_PRECONDITION_FAILED
    assert machine.outcome is EpisodeOutcome.FAILED


def test_legal_handoff_enables_only_arm_b() -> None:
    machine = _machine()
    _enter_b_only(machine)

    assert machine.token is ControlToken.B_ONLY
    machine.require_arm_action("Arm_B")

    with pytest.raises(CollectionStateError) as caught:
        machine.require_arm_action("Arm_A")

    assert caught.value.code is V2FailureCode.INACTIVE_ARM_ACTION


def test_success_requires_stable_finished_bin_and_arm_b_clear() -> None:
    machine = _machine()
    _enter_b_only(machine)

    machine.complete(
        bin_at_finished=True,
        bin_stable=True,
        arm_b_gripper_open=True,
        arm_b_clear=True,
    )

    assert machine.token is ControlToken.NONE
    assert machine.outcome is EpisodeOutcome.SUCCEEDED
    assert machine.failure_code is None


def test_offline_gt_can_gate_provisional_success() -> None:
    machine = _machine()
    _enter_b_only(machine)
    machine.complete(
        bin_at_finished=True,
        bin_stable=True,
        arm_b_gripper_open=True,
        arm_b_clear=True,
    )

    machine.fail_offline_gt(V2FailureCode.P01_TERMINAL_DRIFT_EXCEEDED)

    assert machine.outcome is EpisodeOutcome.FAILED
    assert machine.failure_code is V2FailureCode.P01_TERMINAL_DRIFT_EXCEEDED
    assert machine.token is ControlToken.NONE


@pytest.mark.parametrize(
    ("confirmed", "outcome", "failure_code"),
    [
        (
            True,
            EpisodeOutcome.SAFE_STOPPED,
            V2FailureCode.USER_SAFE_STOP,
        ),
        (
            False,
            EpisodeOutcome.SAFE_STOP_FAILED,
            V2FailureCode.SAFE_STOP_CONFIRMATION_FAILED,
        ),
    ],
)
def test_safe_stop_outcome_is_truthful(
    confirmed: bool,
    outcome: EpisodeOutcome,
    failure_code: V2FailureCode,
) -> None:
    machine = _machine()

    machine.safe_stop(confirmed=confirmed)

    assert machine.token is ControlToken.NONE
    assert machine.outcome is outcome
    assert machine.failure_code is failure_code


def test_p01_profile_ends_after_arm_a_without_handoff() -> None:
    machine = _p01_machine()

    machine.require_arm_action("Arm_A")
    machine.record_part_placement(part_id="P01", slot_id="S11", stable=True)
    machine.complete_p01(arm_a_gripper_open=True, arm_a_clear=True)

    assert machine.placed_parts == ("P01",)
    assert machine.next_part_id is None
    assert machine.token is ControlToken.NONE
    assert machine.outcome is EpisodeOutcome.SUCCEEDED


def test_p01_profile_rejects_arm_b_and_other_parts() -> None:
    machine = _p01_machine()
    with pytest.raises(CollectionStateError) as caught:
        machine.require_arm_action("Arm_B")
    assert caught.value.code is V2FailureCode.INACTIVE_ARM_ACTION

    machine = _p01_machine()
    with pytest.raises(CollectionStateError) as caught:
        machine.record_part_placement(part_id="P02", slot_id="S21", stable=True)
    assert caught.value.code is V2FailureCode.PART_ORDER_VIOLATION


def test_p01_profile_requires_release_and_clearance() -> None:
    machine = _p01_machine()
    machine.record_part_placement(part_id="P01", slot_id="S11", stable=True)

    with pytest.raises(CollectionStateError) as caught:
        machine.complete_p01(arm_a_gripper_open=False, arm_a_clear=True)

    assert caught.value.code is V2FailureCode.FINISH_PRECONDITION_FAILED
    assert machine.outcome is EpisodeOutcome.FAILED
