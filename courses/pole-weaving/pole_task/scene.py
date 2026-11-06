"""Compose the source Q1 robot, soccer-field presentation, and course poles.

The robot subtree, joint/actuator contracts, and floor physics come from the
registered source task 7.7 robot XML, so the pole task cannot silently inherit
a different robot model from the soccer scene.  The soccer scene contributes
only presentation: field markings, goals, lights, materials, and a static,
non-colliding ball.  Three source-geometry cylinders are the only collidable
course objects besides the robot and floor.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from . import ROOT
from .data import CourseCase, load_course
from .source import source_robot_xml

SOURCE_SCENE = ROOT / "expert/resources/q1_description/q1_soccer.xml"
SOURCE_MANIFEST = ROOT / "pole_task/source_manifest.json"
ASSET_ROOT = SOURCE_SCENE.parent

SOURCE_POLE_RADIUS_M = 0.04
SOURCE_POLE_HEIGHT_M = 1.2
SOURCE_POLE_FRICTION = (0.8, 0.005, 0.0001)
FIELD_PRESENTATION_GEOMS = (
    "field_midline",
    "goal_left_post",
    "goal_right_post",
    "goal_back",
    "opponent_goal_left_post",
    "opponent_goal_right_post",
    "opponent_goal_back",
    "floor",
    "soccer_ball",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_asset_paths(root: ET.Element, base: Path) -> None:
    for asset in root.findall(".//asset/*"):
        value = asset.get("file")
        if value and not Path(value).is_absolute():
            asset.set("file", str((base / value).resolve()))


def _import_presentation(document: ET.Element, soccer: ET.Element) -> None:
    """Copy soccer scenery into the source-robot document without physics."""
    soccer_asset = soccer.find("asset")
    document_asset = document.find("asset")
    if soccer_asset is None or document_asset is None:
        raise ValueError("Both source robot and soccer scene must define assets")
    for asset in soccer_asset:
        name = asset.get("name", "")
        if name.startswith("visual_"):
            document_asset.append(asset)

    source_worlds = document.findall("worldbody")
    soccer_world = soccer.find("worldbody")
    if not source_worlds or soccer_world is None:
        raise ValueError("Both source robot and soccer scene must define a worldbody")
    source_world = source_worlds[0]
    for child in soccer_world:
        if child.tag == "body" and child.get("name") == "pelvis":
            continue
        if child.tag == "body" and child.get("name") == "ball":
            geom = child.find("geom[@name='soccer_ball']")
            if geom is None:
                raise ValueError("Soccer scene is missing its scenery ball geom")
            ball = ET.SubElement(
                source_world,
                "geom",
                pos=child.get("pos", "0 0 0"),
                **{key: value for key, value in geom.attrib.items() if key != "name"},
            )
            ball.set("name", "soccer_ball")
            ball.set("contype", "0")
            ball.set("conaffinity", "0")
            continue
        if child.tag == "geom" and child.get("name") == "floor":
            # Keep the source floor's collision and friction; only restyle it.
            continue
        presentation = ET.SubElement(source_world, child.tag, dict(child.attrib))
        presentation[:] = [ET.fromstring(ET.tostring(grandchild)) for grandchild in child]
        if presentation.tag == "geom":
            presentation.set("contype", "0")
            presentation.set("conaffinity", "0")

    source_floor = document.find(".//worldbody/geom[@name='floor']")
    if source_floor is None:
        raise ValueError("Source robot XML is missing its floor geom")
    source_floor.set("material", "visual_turf")
    source_floor.set("rgba", "1 1 1 1")

    source_visual = document.find("visual")
    soccer_visual = soccer.find("visual")
    if source_visual is not None and soccer_visual is not None:
        document.remove(source_visual)
        document.append(soccer_visual)


def build_course_scene(
    case: CourseCase,
    output: Path,
    *,
    robot_xml: Path | None = None,
) -> Path:
    """Derive a per-case scene without mutating the soccer scene on disk."""
    robot_xml = (
        Path(robot_xml).resolve()
        if robot_xml is not None
        else source_robot_xml(SOURCE_MANIFEST)
    )
    tree = ET.parse(robot_xml)
    document = tree.getroot()
    compiler = document.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((robot_xml.parent.parent / "meshes").resolve()))

    soccer = ET.parse(SOURCE_SCENE).getroot()
    _absolute_asset_paths(soccer, ASSET_ROOT)
    _import_presentation(document, soccer)

    world = document.find("worldbody")
    geometry = {
        "type": "cylinder",
        "size": f"{SOURCE_POLE_RADIUS_M} {SOURCE_POLE_HEIGHT_M / 2}",
        "rgba": "0.92 0.72 0.18 1",
        "contype": "1",
        "conaffinity": "1",
        "condim": "3",
        "friction": " ".join(str(value) for value in SOURCE_POLE_FRICTION),
        "density": "0",
        "group": "2",
    }
    for index, (x, y) in enumerate(case.poles_xy, start=1):
        ET.SubElement(
            world,
            "geom",
            name=f"course_pole_{index}",
            pos=f"{x:.9g} {y:.9g} {SOURCE_POLE_HEIGHT_M / 2}",
            **geometry,
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def verify_course_scene(model: mujoco.MjModel, case: CourseCase) -> None:
    """Fail if the composed scene breaks the source robot or pole contract."""
    if (model.nu, model.nq, model.nv) != (22, 29, 28):
        raise ValueError(
            f"Course scene must keep the source Q1 22/29/28 contract, got "
            f"{model.nu}/{model.nq}/{model.nv}"
        )
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Course scene must contain exactly one free robot joint")
    if model.nbody != 27:
        raise ValueError(
            "Course scene must contain only the 26-link source Q1 plus world; "
            f"got {model.nbody} bodies"
        )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") != -1:
        raise ValueError("Soccer ball must be static scenery, not a dynamic body")

    joints = model.actuator_trnid[:, 0]
    if len(set(int(index) for index in joints)) != 22:
        raise ValueError("Course scene must map one actuator to each Q1 hinge")
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Course actuators must use joint transmissions")
    if np.any(model.actuator_gear[:, 1:] != 0) or np.any(model.actuator_gear[:, 0] == 0):
        raise ValueError("Course actuators must keep the source unit gearing")
    if not np.all(model.actuator_ctrllimited):
        raise ValueError("Course actuators must keep source control limits")

    for name in FIELD_PRESENTATION_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            raise ValueError(f"Course scene is missing soccer presentation geom {name!r}")
        if name in ("floor", "soccer_ball"):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                if name == "soccer_ball":
                    raise ValueError("Soccer ball presentation must not collide")
        elif model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
            raise ValueError(f"Presentation geom {name!r} must not collide")

    for index, (x, y) in enumerate(case.poles_xy, start=1):
        name = f"course_pole_{index}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            raise ValueError(f"Course scene is missing pole {name!r}")
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError(f"{name} must be a cylinder")
        size = model.geom_size[geom_id]
        if not np.allclose(
            size[:2], [SOURCE_POLE_RADIUS_M, SOURCE_POLE_HEIGHT_M / 2], atol=1e-6
        ):
            raise ValueError(f"{name} has unexpected source pole dimensions")
        pos = model.geom_pos[geom_id]
        if not np.allclose(pos[:2], [x, y], atol=1e-6):
            raise ValueError(f"{name} does not match its paired course pole")
        if not np.isclose(pos[2], SOURCE_POLE_HEIGHT_M / 2, atol=1e-6):
            raise ValueError(f"{name} is not centered on the source pole height")
        if (
            model.geom_contype[geom_id] != 1
            or model.geom_conaffinity[geom_id] != 1
            or model.geom_condim[geom_id] != 3
            or not np.allclose(model.geom_friction[geom_id], SOURCE_POLE_FRICTION)
        ):
            raise ValueError(f"{name} differs from the source pole collision contract")

    collidable = [
        int(geom_id)
        for geom_id in np.flatnonzero(
            (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        if model.geom_bodyid[geom_id] == 0
    ]
    expected = {"floor"} | {
        f"course_pole_{index}" for index in range(1, len(case.poles_xy) + 1)
    }
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in collidable
    }
    if actual != expected:
        raise ValueError(
            f"Only source floor and course poles may collide in the field scene; got {sorted(actual)}"
        )


def nominal_scene_path(output: Path | None = None) -> Path:
    if output is None:
        output = ROOT / "pole_task/runs/scene/q1_soccer_pole_course.xml"
    return build_course_scene(load_course("nominal-01"), output)
