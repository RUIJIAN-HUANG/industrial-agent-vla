"""Three-camera Isaac Replicator to CAS observation pipeline."""

from __future__ import annotations

from typing import Any, Mapping

from simulation.rgb_cas_bridge import IsaacRgbCasPublisher

FROZEN_CAMERA_STREAMS = {
    "CAM_A_TOP": "arm_a_rgb",
    "CAM_HANDOFF": "handoff_rgb",
    "CAM_B_TOP": "arm_b_rgb",
}


def build_camera_payload(
    references: Mapping[str, Mapping[str, Any]], active_arm: str
) -> dict[str, Any]:
    """Build the schema-frozen camera object from three fresh CAS refs."""

    missing = set(FROZEN_CAMERA_STREAMS) - set(references)
    if missing:
        raise ValueError(f"missing frozen camera references: {sorted(missing)}")
    if active_arm == "Arm_A":
        full_camera = "CAM_A_TOP"
    elif active_arm == "Arm_B":
        full_camera = "CAM_B_TOP"
    elif active_arm == "NONE":
        full_camera = "CAM_HANDOFF"
    else:
        raise ValueError(f"unsupported active arm: {active_arm!r}")
    return {
        "full_image": dict(references[full_camera]),
        **{
            stream: dict(references[camera_id])
            for camera_id, stream in FROZEN_CAMERA_STREAMS.items()
        },
        "wrist_image": None,
    }


class IsaacRgbObservationPipeline:
    """Own three RenderProducts and publish synchronized RGB CAS references."""

    def __init__(
        self,
        *,
        simulation_app: Any,
        scene_config: Mapping[str, Any],
        publisher: IsaacRgbCasPublisher,
        rep_module: Any | None = None,
        warmup_updates: int = 8,
        rt_subframes: int = 1,
    ) -> None:
        if set(publisher.cameras) != set(FROZEN_CAMERA_STREAMS):
            raise ValueError("teleop smoke requires exactly the three frozen cameras")
        if rep_module is None:
            import omni.replicator.core as rep_module  # type: ignore[import-not-found]

        self._rep = rep_module
        self._simulation_app = simulation_app
        self._publisher = publisher
        self._rt_subframes = int(rt_subframes)
        self._resources: dict[str, tuple[Any, Any]] = {}
        camera_config = {
            str(item["id"]): item
            for item in scene_config.get("cameras", [])
            if isinstance(item, Mapping) and "id" in item
        }
        for camera_id, spec in publisher.cameras.items():
            raw = camera_config.get(camera_id)
            if raw is None:
                raise ValueError(f"missing scene camera for {camera_id}")
            prim_path = raw.get("prim_path", f"/World/Cameras/{camera_id}")
            if not isinstance(prim_path, str) or not prim_path:
                raise ValueError(f"invalid camera prim_path for {camera_id}")
            render_product = self._rep.create.render_product(
                prim_path,
                (spec.width, spec.height),
                name=f"{camera_id}_teleop_rgb",
            )
            annotator = self._rep.annotators.get("rgb")
            annotator.attach(render_product)
            self._resources[camera_id] = (render_product, annotator)
        for _ in range(max(0, int(warmup_updates))):
            self._simulation_app.update()

    def capture_references(self) -> dict[str, Mapping[str, Any]]:
        """Publish one synchronized three-camera set to CAS."""

        self._rep.orchestrator.step(rt_subframes=self._rt_subframes)
        references: dict[str, Mapping[str, Any]] = {}
        for camera_id, (_, annotator) in self._resources.items():
            reference = self._publisher.publish(camera_id, annotator.get_data())
            references[camera_id] = reference.to_dict()
        return references

    def capture(self, active_arm: str) -> dict[str, Any]:
        return build_camera_payload(self.capture_references(), active_arm)

    def close(self) -> None:
        resources, self._resources = self._resources, {}
        for render_product, annotator in resources.values():
            annotator.detach()
            render_product.destroy()
        if resources:
            self._rep.orchestrator.wait_until_complete()
