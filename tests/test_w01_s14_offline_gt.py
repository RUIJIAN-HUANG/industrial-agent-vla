from simulation.offline_gt import w01_s14_task_pass


def test_w01_s14_requires_the_target_cell_but_not_orientation() -> None:
    assert w01_s14_task_pass(nearest_slot_id="S14", center_inside_target_cell=True)
    assert not w01_s14_task_pass(nearest_slot_id="S24", center_inside_target_cell=True)
    assert not w01_s14_task_pass(nearest_slot_id="S14", center_inside_target_cell=False)
