"""Process-local collision capacity configuration before Warp graph construction."""
from importlib import import_module


class CollisionCapacity:
    minimum_horizon=128

    @classmethod
    def configure(cls,epa_iterations):
        # Reserve a larger workspace for intersecting robot meshes. Update
        # imported aliases too: each collider allocates its own workspace.
        modules=[import_module('mujoco_warp._src.'+name)
                 for name in ('types','collision_convex','collision_flex')]
        horizon=max(cls.minimum_horizon,10+2*int(epa_iterations),
                    *(module.MJ_MAX_EPAHORIZON for module in modules))
        for module in modules:module.MJ_MAX_EPAHORIZON=horizon
        return dict(epa_horizon=horizon,ccd_iterations=int(epa_iterations))
