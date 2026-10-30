"""Versioned curriculum gates and fixed-width task commands."""
from collections import deque
COURSES = ('balance', 'balance_push', 'move', 'backward', 'lateral', 'turn', 'stop',
           'approach', 'align', 'kick_left', 'kick_right', 'side_left', 'side_right',
           'kick_recover', 'dribble', 'pass_receive', 'goalkeeper', 'duel', 'attack_2v1',
           'game_2v2', 'teamplay', 'match')
LOW_LEVEL = set(COURSES[:7])
KICKS = {'kick_left', 'kick_right', 'side_left', 'side_right', 'kick_recover'}
DURATIONS = {s: (300 if s in LOW_LEVEL else 600 if s not in ('teamplay','match') else 1500) for s in COURSES}
FAMILIES = {s: 'balance' if s in ('balance','balance_push') else 'move' if s in LOW_LEVEL else
            'approach' if s in ('approach','align') else 'kick' if s in KICKS else
            'match' if s == 'match' else 'teamplay' for s in COURSES}
AMPLITUDE = {s: (.12 if s == 'balance' else .18 if s == 'balance_push' else .35 if s in LOW_LEVEL
                 else .55 if s in ('approach','align') else .75 if s in KICKS else 1.) for s in COURSES}
# Independent task identity encoded in existing six command channels, preserving network dimensions.
def command(stage):
    old = ('balance','move','approach','kick','teamplay','match')
    v = [float(i == old.index(FAMILIES[stage])) for i in range(6)]
    if stage not in old:
        v[(old.index(FAMILIES[stage])+1)%6] = (COURSES.index(stage)+1)/32
    return v


class Curriculum:
    version = 2
    def __init__(self, stage='balance'):
        self.index = COURSES.index('kick_left' if stage == 'kick' else stage)
        self.history = deque(maxlen=2048)
        self.monitor = deque(maxlen=2048)
        self.stage_updates = 0
        self.passed_windows = 0
        self.completed_total = 0
        self.last_reason = 'collecting'

    @property
    def stage(self): return COURSES[self.index]

    def record(self, info):
        if info['stage'] == self.stage:
            self.history.append({k: float(info.get(k, 0)) for k in ('stable','success','duration','progress','touch','recover','team')})
            self.monitor.append(self.history[-1])
            self.completed_total += 1

    def metrics(self):
        n = len(self.history)
        mean = lambda k: sum(x[k] for x in self.monitor)/max(len(self.monitor),1)
        teams = [[x for x in self.monitor if x['team']==t] for t in (0,1)]
        team_mean = lambda t,k: sum(x[k] for x in teams[t])/max(len(teams[t]),1)
        return dict(stage=self.stage, stage_index=self.index, completed=n, completed_total=self.completed_total,
                    stage_updates=self.stage_updates, stable_rate=mean('stable'), success_rate=mean('success'),
                    episode_seconds=mean('duration'), progress_m=mean('progress'), foot_touch_rate=mean('touch'),
                    recovery_rate=mean('recover'), passed_windows=self.passed_windows, gate_reason=self.last_reason,
                    red_success=team_mean(0,'success'), blue_success=team_mean(1,'success'),
                    red_stability=team_mean(0,'stable'),blue_stability=team_mean(1,'stable'))

    def advance(self):
        self.stage_updates += 1
        if self.stage_updates % 16: return False
        m = self.metrics()
        required = .85 if self.stage == 'balance' else .7 if self.stage in LOW_LEVEL else .5 if self.stage in ('approach','align') else .35
        # Both teams must pass separately; aggregate success cannot hide a failed team.
        cohorts = [[x for x in self.history if int(x['team'])==t] for t in (0,1)]
        team_ready = all(len(x)>=256 for x in cohorts)
        team_pass = all(sum(x['stable'] for x in group)/max(len(group),1)>=.85 and
                        sum(x['success'] for x in group)/max(len(group),1)>=required for group in cohorts)
        # Disjoint windows, minimum sample count, stability AND task competence.
        ready = self.stage_updates >= 32 and len(self.history) >= 512 and team_ready
        passed = ready and team_pass
        self.passed_windows = self.passed_windows+1 if passed else 0
        self.last_reason = 'pass' if passed else 'samples' if not ready else 'stability' if m['stable_rate'] < .85 else 'task'
        if ready: self.history.clear()
        if self.passed_windows >= 3 and self.index < len(COURSES)-1:
            self.index += 1; self.stage_updates = 0; self.history.clear(); self.monitor.clear(); self.passed_windows = 0
            self.completed_total = 0; self.last_reason = 'collecting'
            return True
        return False

    def state_dict(self):
        return dict(version=2,stage=self.stage,index=self.index,history=list(self.history),monitor=list(self.monitor),stage_updates=self.stage_updates,
                    passed_windows=self.passed_windows,completed_total=self.completed_total,last_reason=self.last_reason)

    def load_state_dict(self, state):
        if state.get('version') != 2:
            old = ('balance','move','approach','kick','teamplay','match')[state['index']]
            # New prerequisites must be verified; retain policy but recertify from balance.
            self.__init__('balance'); self.last_reason = 'legacy_checkpoint_recertification'; return
        self.index = COURSES.index(state['stage']); self.history = deque(state['history'],maxlen=2048)
        self.monitor = deque(state.get('monitor',state['history']),maxlen=2048)
        for k in ('stage_updates','passed_windows','completed_total','last_reason'): setattr(self,k,state[k])
