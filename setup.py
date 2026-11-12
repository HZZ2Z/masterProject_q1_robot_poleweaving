from pathlib import Path

from setuptools import setup
from setuptools.command.develop import develop


class DevelopInPlace(develop):
    """Expose each canonical source root in legacy editable installs."""

    def run(self):
        super().run()
        project_root = Path(__file__).resolve().parent
        source_roots = (
            "courses/pole-weaving",
            "shared/supermimic",
            "shared/q1-mujoco",
        )
        site_packages = Path(self.install_dir)
        site_packages.mkdir(parents=True, exist_ok=True)
        (site_packages / "q1_robot_soccer_sources.pth").write_text(
            "\n".join(str(project_root / path) for path in source_roots) + "\n",
            encoding="utf-8",
        )


setup(
    name="q1-robot-soccer",
    version="0.1.0",
    description="Course and shared Python modules for Q1 Robot Soccer",
    packages=["q1_supermimic", "pole_task", "q1_soccer", "expert", "envs", "utils"],
    package_dir={
        "": "shared/q1-mujoco",
        "q1_supermimic": "courses/pole-weaving/q1_supermimic",
        "pole_task": "courses/pole-weaving/pole_task",
        "q1_soccer": "shared/supermimic/q1_soccer",
        "expert": "shared/supermimic/expert",
        "envs": "shared/q1-mujoco/envs",
        "utils": "shared/q1-mujoco/utils",
    },
    py_modules=["play_mujoco", "evaluate_q1"],
    cmdclass={"develop": DevelopInPlace},
)
