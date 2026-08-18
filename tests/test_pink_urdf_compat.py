from __future__ import annotations

import xml.etree.ElementTree as ET

from simulation.pink_urdf_compat import prepare_pink_compatible_urdf


def test_renames_only_fixed_joint_matching_its_child_link(tmp_path):
    source = tmp_path / "franka.urdf"
    source.write_text(
        """<robot name="panda">
  <link name="base" />
  <link name="panda_hand" />
  <link name="panda_leftfingertip" />
  <joint name="panda_joint7" type="revolute">
    <parent link="base" /><child link="panda_hand" />
  </joint>
  <joint name="panda_leftfingertip" type="fixed">
    <parent link="panda_hand" /><child link="panda_leftfingertip" />
  </joint>
</robot>""",
        encoding="utf-8",
    )

    result, renamed = prepare_pink_compatible_urdf(
        source, output_root=tmp_path / "generated"
    )

    assert renamed == (
        ("panda_leftfingertip", "panda_leftfingertip__pink_fixed_joint"),
    )
    assert source.read_text(encoding="utf-8").count("panda_leftfingertip") == 3
    root = ET.parse(result).getroot()
    joints = {joint.get("name"): joint.get("type") for joint in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    assert joints["panda_joint7"] == "revolute"
    assert joints["panda_leftfingertip__pink_fixed_joint"] == "fixed"
    assert "panda_leftfingertip" in links


def test_returns_original_when_no_conflicting_fixed_joint_exists(tmp_path):
    source = tmp_path / "clean.urdf"
    source.write_text(
        """<robot name="clean"><link name="base" /><link name="tool" />
<joint name="tool_mount" type="fixed"><parent link="base" />
<child link="tool" /></joint></robot>""",
        encoding="utf-8",
    )

    result, renamed = prepare_pink_compatible_urdf(source)

    assert result == str(source.resolve())
    assert renamed == ()
