"""Versioned curriculum gates and fixed-width task commands."""
from collections import deque
from itertools import zip_longest
COURSES = ('balance','balance_push','weight_shift','step_in_place','slow_walk','move','backward','lateral',
           'velocity_track','turn','walk_turn','stop','start_stop','approach','align','kick_left','kick_right',
           'side_left','side_right','kick_recover','dribble','pass_receive','goalkeeper','duel','attack_2v1',
           'game_2v2','teamplay','match')
LEGACY_COURSES = ('balance','balance_push','move','backward','lateral','turn','stop','approach','align',
                  'kick_left','kick_right','side_left','side_right','kick_recover','dribble','pass_receive',
                  'goalkeeper','duel','attack_2v1','game_2v2','teamplay','match')
LOW_LEVEL = set(COURSES[:13])
GAIT_COURSES = set(COURSES[3:13]) | {'approach'}
KICKS = {'kick_left','kick_right','side_left','side_right','kick_recover'}
_SECONDS = (8,8,10,12,12,12,12,12,15,12,15,12,18,12,15,12,12,12,12,15,18,18,12,20,20,25,30,30)
_THRESHOLDS = (.95,.90,.80,.80,.80,.80,.75,.75,.75,.75,.75,.80,.80,.70,.65,.50,.50,.45,.45,.60,.50,.45,.50,.35,.45,.35,.40,.0)
DURATIONS = {s:int(t/.02) for s,t in zip(COURSES,_SECONDS)}
SUCCESS = dict(zip(COURSES,_THRESHOLDS))
FAMILIES = {s:'balance' if s in ('balance','balance_push','weight_shift') else 'move' if s in LOW_LEVEL else
            'approach' if s in ('approach','align') else 'kick' if s in KICKS else
            'match' if s=='match' else 'teamplay' for s in COURSES}
AMPLITUDE = {s:(.12 if s=='balance' else .18 if s in ('balance_push','weight_shift') else .45 if s in LOW_LEVEL
                 else .55 if s in ('approach','align') else .75 if s in KICKS else 1.) for s in COURSES}
def command(stage):
    old=('balance','move','approach','kick','teamplay','match')
    v=[float(i==old.index(FAMILIES[stage])) for i in range(6)]
    if stage not in old:
        v[(old.index(FAMILIES[stage])+1)%6] = ((LEGACY_COURSES.index(stage)+1)/32 if stage in LEGACY_COURSES
                                           else 1+(COURSES.index(stage)+1)/32)
    return v


class TeamWindow:
    """Bounded FIFO per team; completion order cannot evict the other team."""
    def __init__(self, rows=()):
        self.teams = [deque(maxlen=1024), deque(maxlen=1024)]
        for row in rows:
            self.append(row)

    def append(self, row):
        team = int(row['team'])
        if team not in (0, 1):
            raise ValueError('Curriculum team must be red (0) or blue (1)')
        self.teams[team].append(row)

    def __len__(self):
        return sum(map(len, self.teams))

    def __iter__(self):
        for pair in zip_longest(*self.teams):
            for row in pair:
                if row is not None:
                    yield row

    def clear(self):
        for team in self.teams:
            team.clear()


class Curriculum:
    version = 3
    def __init__(self, stage='balance'):
        self.index = COURSES.index('kick_left' if stage == 'kick' else stage)
        self.history = TeamWindow()
        self.monitor = TeamWindow()
        self.stage_updates = 0
        self.passed_windows = 0
        self.completed_total = 0
        self.last_reason = 'collecting'

    @property
    def stage(self): return COURSES[self.index]

    def record(self, info):
        if info['stage'] == self.stage:
            row = {k: float(info.get(k, 0)) for k in ('stable','success','duration','progress','touch','recover','team')}
            self.history.append(row)
            self.monitor.append(row)
            self.completed_total += 1

    def metrics(self):
        n = len(self.history)
        mean = lambda k: sum(x[k] for x in self.monitor)/max(len(self.monitor),1)
        teams = [[x for x in self.monitor if x['team']==t] for t in (0,1)]
        team_mean = lambda t,k: sum(x[k] for x in teams[t])/len(teams[t]) if teams[t] else None
        return dict(stage=self.stage, stage_index=self.index, completed=n, completed_total=self.completed_total,
                    stage_updates=self.stage_updates, stable_rate=mean('stable'), success_rate=mean('success'),
                    episode_seconds=mean('duration'), progress_m=mean('progress'), foot_touch_rate=mean('touch'),
                    recovery_rate=mean('recover'), passed_windows=self.passed_windows, gate_reason=self.last_reason,
                    red_success=team_mean(0,'success'), blue_success=team_mean(1,'success'),
                    red_stability=team_mean(0,'stable'),blue_stability=team_mean(1,'stable'),
                    red_samples=len(teams[0]), blue_samples=len(teams[1]),
                    red_gate_samples=len(self.history.teams[0]),blue_gate_samples=len(self.history.teams[1]))

    def advance(self, promote=True):
        self.stage_updates += 1
        if self.stage_updates % 16: return False
        m = self.metrics()
        required = SUCCESS[self.stage]
        # Both teams must pass separately; aggregate success cannot hide a failed team.
        cohorts = [[x for x in self.history if int(x['team'])==t] for t in (0,1)]
        team_ready = all(len(x)>=256 for x in cohorts)
        team_pass = all(sum(x['stable'] for x in group)/max(len(group),1)>=.90 and
                        sum(x['success'] for x in group)/max(len(group),1)>=required for group in cohorts)
        # Disjoint windows, minimum sample count, stability AND task competence.
        ready = self.stage_updates >= 32 and len(self.history) >= 512 and team_ready
        passed = ready and team_pass
        self.passed_windows = self.passed_windows+1 if passed else 0
        stable = all(sum(x['stable'] for x in group)/max(len(group),1)>=.90 for group in cohorts)
        self.last_reason = 'pass' if passed else 'samples' if not ready else 'stability' if not stable else 'task'
        if ready: self.history.clear()
        if self.passed_windows >= 3 and not promote:
            return True
        if self.passed_windows >= 3 and self.index < len(COURSES)-1:
            self.index += 1; self.stage_updates = 0; self.history.clear(); self.monitor.clear(); self.passed_windows = 0
            self.completed_total = 0; self.last_reason = 'collecting'
            return True
        return False

    def state_dict(self):
        return dict(version=3,window_policy='per_team',stage=self.stage,index=self.index,history=list(self.history),monitor=list(self.monitor),stage_updates=self.stage_updates,
                    passed_windows=self.passed_windows,completed_total=self.completed_total,last_reason=self.last_reason)

    def load_state_dict(self, state):
        if state.get('version') != 3:
            # New prerequisites must be verified; retain policy but recertify from balance.
            self.__init__('balance'); self.last_reason = 'legacy_checkpoint_recertification'; return
        self.index = COURSES.index(state['stage']); self.history = TeamWindow(state['history'])
        self.monitor = TeamWindow(state.get('monitor',state['history']))
        for k in ('stage_updates','passed_windows','completed_total','last_reason'): setattr(self,k,state[k])

    def recertify(self, reason):
        """Keep the current course; collect fresh evidence under changed rewards."""
        self.history.clear(); self.monitor.clear()
        self.stage_updates = 0; self.passed_windows = 0; self.completed_total = 0
        self.last_reason = reason
