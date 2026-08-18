"""Create a project-local URDF view compatible with Pinocchio reduction.

Some vendor Franka URDFs give a fixed joint the same name as its child link.
Pinocchio represents those as a ``FIXED_JOINT`` frame and a ``BODY`` frame
with the same name. That is legal, but Pinocchio 2.7 cannot reduce the model
without an explicit frame type. Pink does not expose that choice, so rename
only the conflicting fixed-joint declarations in a temporary URDF copy.
"""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def prepare_pink_compatible_urdf(
    source_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return a Pink-compatible URDF path and the fixed joints renamed.

    The source file is never modified. Link names, movable-joint names, TCP
    frames and mesh references are preserved exactly.
    """

    source = Path(source_path).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    existing_joint_names = {
        str(joint.get("name", "")) for joint in root.findall("joint")
    }
    renamed: list[tuple[str, str]] = []

    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        child = joint.find("child")
        old_name = str(joint.get("name", ""))
        child_link = "" if child is None else str(child.get("link", ""))
        if not old_name or old_name != child_link:
            continue

        base_name = f"{old_name}__pink_fixed_joint"
        new_name = base_name
        suffix = 2
        while new_name in existing_joint_names:
            new_name = f"{base_name}_{suffix}"
            suffix += 1
        joint.set("name", new_name)
        existing_joint_names.add(new_name)
        renamed.append((old_name, new_name))

    if not renamed:
        return str(source), ()

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(xml_bytes).hexdigest()[:16]
    destination_root = (
        Path(output_root)
        if output_root is not None
        else Path(tempfile.gettempdir()) / "industrial-agent-vla-pink-urdf"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{source.stem}-pink-{digest}.urdf"
    if not destination.exists() or destination.read_bytes() != xml_bytes:
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(xml_bytes)
        temporary.replace(destination)
    return str(destination), tuple(renamed)
