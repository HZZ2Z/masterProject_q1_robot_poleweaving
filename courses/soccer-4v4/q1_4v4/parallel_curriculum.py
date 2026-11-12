"""Prerequisite-gated multi-course sampling with independent evidence windows."""
import math
from .course_config import COURSES, Curriculum

PREREQUISITES = {
 'balance':(), 'balance_push':('balance',), 'weight_shift':('balance',),
 'step_in_place':('weight_shift',), 'slow_walk':('step_in_place',),
 'move':('slow_walk',), 'backward':('slow_walk',), 'lateral':('slow_walk',),
 'velocity_track':('move','backward','lateral'), 'turn':('step_in_place',),
 'walk_turn':('slow_walk','turn'), 'stop':('slow_walk',), 'start_stop':('slow_walk','stop'),
 'approach':('move','turn'), 'align':('approach',),
 'kick_left':('align','weight_shift'), 'kick_right':('align','weight_shift'),
 'side_left':('kick_left',), 'side_right':('kick_right',),
 'kick_recover':('kick_left','kick_right'), 'dribble':('approach','kick_left','kick_right','stop'),
 'pass_receive':('dribble',), 'goalkeeper':('lateral','stop','kick_left','kick_right'),
 'duel':('dribble','kick_recover'), 'attack_2v1':('duel','pass_receive'),
 'game_2v2':('attack_2v1','goalkeeper'), 'teamplay':('game_2v2','pass_receive'),
 'match':('teamplay','goalkeeper'),
}


class ParallelCurriculum:
    def __init__(self):
        self.courses={name:Curriculum(name) for name in COURSES}
        self.mastered=set()
        self.last_allocations={}

    @property
    def ready(self):
        return [name for name in COURSES if name not in self.mastered
                and set(PREREQUISITES[name])<=self.mastered]

    @property
    def frontier(self):
        ready=set(self.ready)
        return [name for name in COURSES if name not in self.mastered|ready
                and set(PREREQUISITES[name])<=self.mastered|ready]

    @property
    def stage(self):return self.ready[0] if self.ready else 'match'

    @property
    def index(self):return COURSES.index(self.stage)

    def weights(self):
        ready=self.ready; review=[s for s in COURSES if s in self.mastered]; frontier=self.frontier
        # Until balance is certified, allocate everything to balance.
        if 'balance' not in self.mastered:return {'balance':1.}
        fractions=[(ready, .8),(review,.1),(frontier,.1)]
        unused=sum(f for group,f in fractions if not group)
        return {name:(fraction+(unused if group is ready else 0))/len(group)
                for group,fraction in fractions if group for name in group}

    def allocation(self,count):
        weights=self.weights();names=[s for s in COURSES if s in weights]
        if count<len(names):raise ValueError('At least one world per sampled course is required')
        # Reserve one world per course, then largest-remainder apportionment.
        remaining=count-len(names); raw={s:remaining*weights[s] for s in names}
        result={s:1+math.floor(raw[s]) for s in names}
        extra=count-sum(result.values())
        order=sorted(names,key=lambda s:-(raw[s]-math.floor(raw[s])))
        for s in order[:extra]:result[s]+=1
        self.last_allocations=result
        return result

    def record(self,info):self.courses[info['stage']].record(info)

    def advance(self):
        before=set(self.ready); changed=False
        for name in COURSES:
            if name in before and name!='match':
                if self.courses[name].advance(promote=False):
                    self.mastered.add(name);changed=True
        for name in set(self.ready)-before:
            self.courses[name].recertify('prerequisites_mastered')
        return changed

    def metrics(self):
        return dict(self.courses[self.stage].metrics(), scheduler='parallel',
                    mastered_count=len(self.mastered), unlocked_count=len(self.mastered|set(self.ready)),
                    mastered=sorted(self.mastered,key=COURSES.index), ready=self.ready, frontier=self.frontier,
                    allocations=dict(self.last_allocations),
                    courses={s:self.courses[s].metrics() for s in COURSES
                             if s in self.mastered|set(self.ready)|set(self.frontier)})

    def state_dict(self):
        return {'kind':'parallel','version':3,'stage':self.stage,'index':self.index,'mastered':sorted(self.mastered,key=COURSES.index),
                'courses':{s:c.state_dict() for s,c in self.courses.items()},
                'allocations':self.last_allocations}

    def load_state_dict(self,state):
        if state.get('kind')=='parallel':
            self.mastered=set(state['mastered'])
            for name,saved in state['courses'].items():self.courses[name].load_state_dict(saved)
            self.last_allocations=dict(state.get('allocations',{}))
        else:
            if state.get('version')!=3 or state.get('stage')=='balance':
                raise ValueError('Parallel continuation requires a checkpoint AFTER course 0 passed')
            index=COURSES.index(state['stage'])
            self.mastered=set(COURSES[:index])
            for name in self.mastered:
                self.courses[name].passed_windows=3
                self.courses[name].last_reason='imported_mastery'
            self.courses[state['stage']].load_state_dict(state)

    def recertify(self,reason):
        for name in self.ready:self.courses[name].recertify(reason)
