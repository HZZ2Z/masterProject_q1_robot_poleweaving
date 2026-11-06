"""Course scene checks: source robot physics plus soccer-field presentation."""

import numpy as np
import mujoco
import pytest

from pole_task.data import load_course
from pole_task.environment import PoleEnvironment
from pole_task.scene import (
    SOURCE_MANIFEST,
    SOURCE_POLE_FRICTION,
    SOURCE_POLE_HEIGHT_M,
    SOURCE_POLE_RADIUS_M,
    build_course_scene,
    verify_course_scene,
)
from pole_task.source import source_robot_xml


def _model(tmp_path, case_id="nominal-01"):
    case = load_course(case_id)
    path = build_course_scene(case, tmp_path / f"{case_id}.xml")
    return case, mujoco.MjModel.from_xml_path(str(path))


def _robot_geoms(model):
    geoms = {}
    for index in range(model.ngeom):
        body = int(model.geom_bodyid[index])
        if body == 0:
            continue
        geoms.setdefault(body, []).append(
            (
                int(model.geom_type[index]),
                tuple(np.round(model.geom_size[index], 9)),
                tuple(np.round(model.geom_pos[index], 9)),
                tuple(np.round(model.geom_quat[index], 9)),
                tuple(np.round(model.geom_friction[index], 9)),
                int(model.geom_contype[index]),
                int(model.geom_conaffinity[index]),
            )
        )
    return {body: sorted(values) for body, values in geoms.items()}


def test_course_scene_embeds_the_registered_source_robot_exactly(tmp_path):
    case, model = _model(tmp_path)
    source = mujoco.MjModel.from_xml_path(str(source_robot_xml(SOURCE_MANIFEST)))

    verify_course_scene(model, case)

    assert (model.nq, model.nv, model.nu, model.nbody) == (29, 28, 22, 27)
    assert [model.body(i).name for i in range(model.nbody)] == [
        source.body(i).name for i in range(source.nbody)
    ]
    assert np.array_equal(model.body_mass, source.body_mass)
    assert np.array_equal(model.body_pos, source.body_pos)
    assert np.array_equal(model.body_inertia, source.body_inertia)
    assert [model.joint(i).name for i in range(model.njnt)] == [
        source.joint(i).name for i in range(source.njnt)
    ]
    assert np.array_equal(model.jnt_range, source.jnt_range)
    assert np.array_equal(model.dof_armature, source.dof_armature)
    assert np.array_equal(model.dof_damping, source.dof_damping)
    assert np.array_equal(model.actuator_trnid, source.actuator_trnid)
    assert np.array_equal(model.actuator_gear, source.actuator_gear)
    assert np.array_equal(model.actuator_ctrlrange, source.actuator_ctrlrange)
    assert _robot_geoms(model) == _robot_geoms(source)


def test_field_goals_and_ball_are_presentation_but_poles_collide(tmp_path):
    case, model = _model(tmp_path)

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") == -1
    ball = model.geom("soccer_ball").id
    assert model.geom_bodyid[ball] == 0
    assert model.geom_contype[ball] == 0 and model.geom_conaffinity[ball] == 0
    for name in ("goal_left_post", "goal_back", "opponent_goal_right_post", "field_midline"):
        geom = model.geom(name).id
        assert model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0

    collidable = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
        if model.geom_bodyid[index] == 0
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    }
    assert collidable == {"floor", "course_pole_1", "course_pole_2", "course_pole_3"}
    for index, (x, y) in enumerate(case.poles_xy, start=1):
        pole = model.geom(f"course_pole_{index}")
        assert np.allclose(pole.size[:2], [SOURCE_POLE_RADIUS_M, SOURCE_POLE_HEIGHT_M / 2])
        assert np.allclose(pole.pos[:2], [x, y])
        assert np.isclose(pole.pos[2], SOURCE_POLE_HEIGHT_M / 2)
        assert np.allclose(pole.friction, SOURCE_POLE_FRICTION)


@pytest.mark.parametrize("case_id", ["nominal-01", "eval-01"])
def test_each_scene_pairs_its_own_course_poles(tmp_path, case_id):
    case, model = _model(tmp_path, case_id)
    verify_course_scene(model, case)


def test_headless_pole_contact_terminates_and_goals_do_not_contact(tmp_path):
    case = load_course("nominal-01")
    environment = PoleEnvironment(case, tmp_path / "environment.xml")
    environment.reset()

    x, y = case.poles_xy[0]
    environment.data.qpos[environment.root_q : environment.root_q + 3] = [x, y, 0.75]
    environment.data.qpos[environment.root_q + 3] = 1.0
    mujoco.mj_forward(environment.model, environment.data)

    _, reward, done, transition = environment.step(np.zeros(22))

    assert environment.collision
    assert done
    assert transition["terminal_cause"] == "pole_collision"
    assert transition["pole_contacts"] == ["course_pole_1"]
    assert reward < 0

    goal = environment.model.geom("goal_left_post")
    environment.data.qpos[environment.root_q : environment.root_q + 3] = [
        *goal.pos[:2],
        0.75,
    ]
    mujoco.mj_forward(environment.model, environment.data)
    goal_id = goal.id
    assert not any(
        goal_id in (contact.geom1, contact.geom2) and contact.dist <= 0
        for contact in environment.data.contact
    )
