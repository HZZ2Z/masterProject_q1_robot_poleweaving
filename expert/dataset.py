"""Expert trajectory schema and outcome rules. No learning or optimizer code."""
import numpy as np

SCHEMA_VERSION=1
NATURAL_TERMINALS={'goal','fall','out'}
TRUNCATIONS={'timeout','collector_limit','interrupted','invalid'}


def observation_history(observations,index,initial_history):
    """Exact newest-first causal history, including original reset padding."""
    if not 0<=index<len(observations):raise IndexError(index)
    frames=len(initial_history)
    recent=observations[max(0,index-frames+1):index+1][::-1]
    if len(recent)<frames:
        recent=np.concatenate((recent,initial_history[1:frames-len(recent)+1]),axis=0)
    return recent.copy()


def outcome(cause,contacts,post_touch_progress,duration,first_touch_time,recovery_seconds=1.):
    """A goal, a verified kick, a timeout and a failure are distinct labels."""
    if cause not in NATURAL_TERMINALS|TRUNCATIONS:raise ValueError(cause)
    touched=bool(np.any(contacts))
    recovered=bool(touched and first_touch_time is not None and
                   duration-first_touch_time>=recovery_seconds and cause not in {'fall','invalid'})
    kick=bool(touched and post_touch_progress>=.1 and recovered)
    verified=bool(cause=='goal' and kick)
    return dict(termination=cause,task_success=cause=='goal',verified_success=verified,
                policy_imitation_eligible=verified,contact=touched,effective_kick=kick,
                recovered_after_touch=recovered,timeout=cause=='timeout',
                failure=cause in {'fall','out','invalid'},
                terminated=cause in NATURAL_TERMINALS,truncated=cause in TRUNCATIONS)


def validate_arrays(arrays,meta):
    t=len(arrays['actions']);nq=meta['nq'];nv=meta['nv'];substeps=meta['substeps']
    expected={'qpos':(t+1,nq),'qvel':(t+1,nv),'qacc_warmstart':(t+1,nv),
       'time':(t+1,),'actions':(t,22),'raw_actions':(t,22),'pd_targets':(t,22),
       'observations':(t,79),'initial_history':(50,79),'ball_observation_queue':(t,20,3),
       'applied_torques':(t,substeps,22),'foot_ball_contact':(t,substeps),
       'other_ball_contact':(t,substeps),'ball_goal_progress_m':(t,),
       'post_touch_goal_progress_m':(t,),'terminated':(t,),'truncated':(t,),
       'episode_end':(t,),'world_model_valid':(t,),'policy_imitation_valid':(t,)}
    if set(arrays)!=set(expected):raise ValueError('Unexpected or missing trajectory fields')
    for key,shape in expected.items():
        if arrays[key].shape!=shape:raise ValueError(f'{key}: expected {shape}, got {arrays[key].shape}')
        if not np.isfinite(arrays[key]).all():raise ValueError(f'Non-finite {key}')
    if t:
        if not np.all(np.diff(arrays['time'])>0):raise ValueError('Non-increasing time/reset crossed')
        np.testing.assert_allclose(np.diff(arrays['time']),meta['control_dt'],atol=1e-10,rtol=0)
        if arrays['episode_end'].sum()!=1 or not arrays['episode_end'][-1]:raise ValueError('Episode boundary incorrect')
        if arrays['terminated'].sum()!=int(meta['terminated']):raise ValueError('Termination label mismatch')
        if arrays['truncated'].sum()!=int(meta['truncated']):raise ValueError('Truncation label mismatch')
        if bool(arrays['policy_imitation_valid'].any())!=meta['policy_imitation_eligible']:
            raise ValueError('Failed/timeout trajectory entered policy imitation')
        if not np.all(arrays['policy_imitation_valid']==meta['policy_imitation_eligible']):
            raise ValueError('Full-task imitation mask inconsistent')
        if arrays['foot_ball_contact'].any()!=meta['contact']:raise ValueError('Contact label mismatch')
        if not arrays['world_model_valid'].all():raise ValueError('Invalid dynamics transition persisted')
        # Observations are current sensors + previous executed action, never
        # a future action or privileged ball state.
        np.testing.assert_array_equal(arrays['observations'][1:,57:79],arrays['actions'][:-1])
    return t
