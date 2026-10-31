"""Recover native cached sensors at their exact MuJoCo substep, without learning."""

import hashlib
from pathlib import Path
import numpy as np
import mujoco
import yaml

from expert.runtime import ROOT as EXPERT_ROOT

SENSOR_FIELDS = (
    "position",
    "orientation",
    "angular-velocity",
    "camera_position",
    "camera_orientation",
)


def native_sensors(arrays, cache_dir):
    identity = hashlib.sha256(b"q1-native-sensor-cache-v1")
    identity.update(
        (EXPERT_ROOT / "resources/q1_description/q1_soccer.xml").read_bytes()
    )
    identity.update((EXPERT_ROOT / "configs/Q1.yaml").read_bytes())
    for name in ["qpos", "qvel", "qacc_warmstart", "applied_torques"]:
        identity.update(arrays[name].tobytes())
    cache = Path(cache_dir) / (identity.hexdigest() + ".npy")
    if cache.exists():
        sensors = np.load(cache, allow_pickle=False)
        if (
            sensors.shape != (len(arrays["actions"]) + 1, 17)
            or not np.isfinite(sensors).all()
        ):
            raise ValueError("Invalid native sensor cache")
        return sensors
    model = mujoco.MjModel.from_xml_path(
        str(EXPERT_ROOT / "resources/q1_description/q1_soccer.xml")
    )
    cfg = yaml.safe_load((EXPERT_ROOT / "configs/Q1.yaml").read_text())
    model.opt.timestep = cfg["sim"]["dt"]
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["qpos"][0]
    data.qvel[:] = arrays["qvel"][0]
    data.time = float(arrays["time"][0])
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = arrays["qacc_warmstart"][0]

    def read():
        return np.concatenate([data.sensor(name).data.copy() for name in SENSOR_FIELDS])

    sensors = [read()]
    for t, torques in enumerate(arrays["applied_torques"]):
        for torque in torques:
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        # Sensor recovery must replay the actual physical path, including all
        # contacts. Never silently accept a diverged replay as ground truth.
        if not np.array_equal(data.qpos, arrays["qpos"][t + 1]) or not np.array_equal(
            data.qvel, arrays["qvel"][t + 1]
        ):
            raise ValueError(
                f"Native sensor recovery physics differs at transition {t}"
            )
        sensors.append(read())
    sensors = np.asarray(sensors, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, sensors, allow_pickle=False)
    temporary.replace(cache)
    return sensors
