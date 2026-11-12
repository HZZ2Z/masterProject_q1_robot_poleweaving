"""Evidence-gated curriculum; failed short episodes count in every gate."""
from collections import deque

STAGES = ('balance', 'move', 'approach', 'kick', 'teamplay', 'match')
AMPLITUDES = (.12, .3, .5, .7, .85, 1.)
# Fixed per-stage metrics, never compare returns between stages.
SUCCESS_THRESHOLDS = (.85, .60, .50, .30, .20)


class Curriculum:
    def __init__(self, stage='balance'):
        self.index = STAGES.index(stage)
        self.history = deque(maxlen=64)
        self.stage_updates = 0

    @property
    def stage(self):
        return STAGES[self.index]

    def record(self, info):
        if info['stage'] == self.stage:
            self.history.append({'stable': bool(info['stable']), 'success': bool(info['success'])})

    def metrics(self):
        n = len(self.history)
        return dict(stage=self.stage, completed=n, stage_updates=self.stage_updates,
                    stable_rate=sum(x['stable'] for x in self.history)/max(n, 1),
                    success_rate=sum(x['success'] for x in self.history)/max(n, 1))

    def advance(self):
        self.stage_updates += 1
        m = self.metrics()
        if self.index < len(STAGES)-1 and self.stage_updates >= 32 and len(self.history) >= 32 \
                and m['stable_rate'] >= .85 and m['success_rate'] >= SUCCESS_THRESHOLDS[self.index]:
            self.index += 1
            self.history.clear()
            self.stage_updates = 0
            return True
        return False

    def state_dict(self):
        return dict(index=self.index, history=list(self.history), stage_updates=self.stage_updates)

    def load_state_dict(self, state):
        self.index = state['index']
        self.history = deque(state['history'], maxlen=64)
        self.stage_updates = state['stage_updates']
