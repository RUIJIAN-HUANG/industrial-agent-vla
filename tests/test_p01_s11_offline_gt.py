from simulation.offline_gt import p01_s11_task_pass


def test_small_flange_overhang_does_not_redefine_slot_success() -> None:
    assert p01_s11_task_pass(
        nearest_slot_id="S11",
        inside_target_cell=True,
        upright=True,
        containment_axis_pass={"x": False, "y": True, "z": True},
    )


def test_wrong_slot_tilt_or_vertical_escape_still_fails() -> None:
    assert not p01_s11_task_pass(
        nearest_slot_id="S12",
        inside_target_cell=True,
        upright=True,
        containment_axis_pass={"x": True, "y": True, "z": True},
    )
    assert not p01_s11_task_pass(
        nearest_slot_id="S11",
        inside_target_cell=True,
        upright=False,
        containment_axis_pass={"x": True, "y": True, "z": True},
    )
    assert not p01_s11_task_pass(
        nearest_slot_id="S11",
        inside_target_cell=True,
        upright=True,
        containment_axis_pass={"x": True, "y": True, "z": False},
    )
