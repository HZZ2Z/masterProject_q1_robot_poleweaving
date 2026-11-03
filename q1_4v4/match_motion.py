"""Contact-based motion constraints; no new observations or prescribed gait phase."""
import torch


class MatchMotion:
    revision = 1
    terms = ('support_slide', 'posture_margin', 'excess_flight', 'upper_motion')
    metrics = tuple(f'motion_{key}_{side}' for key in ('slip', 'flight', 'quality', 'upper_error')
                    for side in ('red', 'blue'))
    ramp_iterations = 64

    def __init__(self, count, device):
        self.enabled = False
        self.start_iteration = self.iteration = 0
        self.air_time = torch.zeros((count, 8), device=device)
        self.strike_time = torch.zeros((count,8,2),device=device)
        self.quality = torch.ones_like(self.air_time)
        self.last_metrics = torch.zeros(len(self.metrics), device=device)
        self.upper_joints = torch.tensor([12,14,15,16,17,18,19,20,21], device=device)

    @property
    def ramp(self):
        return min(1., max(0., (self.iteration-self.start_iteration)/self.ramp_iterations)) if self.enabled else 0.

    def reset_worlds(self, ids):
        self.air_time[ids] = 0; self.quality[ids] = 1
        self.strike_time[ids] = 0

    def reset_agents(self, w, a):
        self.air_time[w, a] = 0; self.quality[w, a] = 1
        self.strike_time[w,a] = 0

    @staticmethod
    def posture_errors(up, height, width, bad_contact):
        # Small inner margins give feedback BEFORE the certification boundary.
        return torch.stack((((.90-up)/.15).clamp(min=0),
            ((.38-height)/.05).clamp(min=0)+((height-.47)/.05).clamp(min=0),
            ((.11-width)/.05).clamp(min=0)+((width-.27)/.10).clamp(min=0),
            bad_contact.float()), -1)

    def evaluate(self, env, up, height, width):
        zero = torch.zeros_like(up)
        if not self.enabled:
            self.quality.fill_(1); self.last_metrics.zero_()
            return {key: zero for key in self.terms}
        contact = env.gait_contact.bool()
        task = env.match_task
        strike = task.ball_skill.strikes
        self.strike_time[:] = torch.where(strike,self.strike_time+env.dt,0.)
        strike = strike & (self.strike_time<=.100001)
        # Productive strikes may slide the striking foot; the supporting foot
        # remains constrained. Mere proximity or a body push grants no exemption.
        support = contact & ~strike
        slip = ((env.foot_slip-.03).clamp(min=0)/.12).square().clamp(max=1)
        slide_cost = (slip*support).sum(-1)/support.sum(-1).clamp(min=1)
        errors = self.posture_errors(up, height, width, env.gait_bad_contact.bool())
        errors[..., 2] *= ~strike.any(-1)  # allow brief leg extension during a real strike
        pose_cost = errors.square().amax(-1).clamp(max=1)
        moving = torch.linalg.vector_norm(env.gait_command[..., :2], dim=-1) > .03
        airborne = ~contact.any(-1) & moving & env.active
        self.air_time[:] = torch.where(airborne, self.air_time+env.dt, 0.)
        flight_cost = ((self.air_time-.06)/.14).clamp(0, 1).square()
        # The reference masks these nine DoFs. Allow normal balancing motion,
        # penalizing only large deviations and rapid upper-body oscillation.
        joints = self.upper_joints
        position = env.qpos[:, env.qidx][:, :, joints]-env.neutral[joints]
        velocity = env.qvel[:, env.vidx][:, :, joints]
        upper_cost = (((position.abs()-.35).clamp(min=0)/.5).square().mean(-1)
                      +((velocity.abs()-2.).clamp(min=0)/4.).square().mean(-1)).clamp(max=1)
        intrinsic = torch.exp(-3*pose_cost-2*slide_cost-2*flight_cost)
        self.quality[:] = 1-self.ramp+self.ramp*intrinsic
        def team_mean(value):
            value = (value*env.active).reshape(env.count, 2, 4)
            active = env.active.reshape(env.count, 2, 4)
            return value.sum((0,2))/active.sum((0,2)).clamp(min=1)
        raw_slip = (env.foot_slip*contact).sum(-1)/contact.sum(-1).clamp(min=1)
        self.last_metrics[:] = torch.cat(tuple(team_mean(x) for x in
            (raw_slip, airborne.float(), self.quality, position.abs().mean(-1))))
        return dict(support_slide=-.06*self.ramp*slide_cost,
                    posture_margin=-.06*self.ramp*pose_cost,
                    excess_flight=-.06*self.ramp*flight_cost,
                    upper_motion=-.02*self.ramp*upper_cost)


def configure_motion(options, checkpoint, iteration):
    """Legacy checkpoints retain their previous reward mode unless opted in."""
    saved = checkpoint.get('motion_constraints', {}) if checkpoint else {}
    selected = getattr(options, 'match_motion_constraints', None)
    if selected is None: selected = saved.get('enabled', False)
    changed = bool(checkpoint and (selected != saved.get('enabled', False) or
                   (selected and saved.get('revision') != MatchMotion.revision)))
    options.match_motion_constraints = bool(selected)
    return dict(enabled=bool(selected), revision=MatchMotion.revision,
                start_iteration=iteration if changed or not saved else saved['start_iteration'],
                ramp_iterations=MatchMotion.ramp_iterations), changed
