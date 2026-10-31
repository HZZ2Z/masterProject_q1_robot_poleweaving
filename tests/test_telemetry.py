import torch
import pytest
from q1_4v4.course_config import COURSES,Curriculum
from q1_4v4.telemetry import active_team_reward,course_metrics,macro_metrics,dashboard_specs


def test_every_dashboard_section_has_six_panels():
    specs=dashboard_specs(COURSES)
    assert len(specs)==len(COURSES)+2
    assert all(len(panels)==6 for _,panels in specs)
    for _,panels in specs:
        assert all(p['keys'] and p['title'] for p in panels)
    assert len(dashboard_specs([],parallel=False,per_course=False))==2
    charts=next(p for name,p in specs if name.endswith('· match'))
    assert len(charts)==6 and not any('目标成功率' in k for p in charts for k in p['keys'])


def test_raw_reward_uses_only_active_participants_and_copies_scalars():
    reward=torch.tensor([[[2.],[100.],[100.],[100.],[4.],[100.],[100.],[100.]]])
    active=torch.tensor([[True,False,False,False,True,False,False,False]])
    result=active_team_reward(reward,active)
    reward.add_(1000)  # timeout bootstrapping later must not mutate telemetry
    torch.testing.assert_close(result,torch.tensor([2.,4.]))
    assert active_team_reward(reward,torch.zeros_like(active)).tolist()==[0.,0.]


def test_unobserved_course_has_missing_quality_not_zero():
    stats=Curriculum('balance').metrics()
    values=course_metrics('balance',stats,(None,None),None,None,0,128)
    for key in ('成功率','稳定率','红队成功率','蓝队成功率','摔倒率','平均回合秒数'):
        assert values['06_课程/balance/'+key] is None
    assert values['06_课程/balance/球场占比']==0


def test_macro_quality_gives_equal_weight_to_tasks_not_sample_counts():
    stats={'balance':dict(red_samples=10000,blue_samples=10000,success_rate=1.,stable_rate=1.),
           'balance_push':dict(red_samples=1,blue_samples=1,success_rate=0.,stable_rate=.2),
           'weight_shift':dict(red_samples=0,blue_samples=0,success_rate=0.,stable_rate=0.)}
    values=macro_metrics(stats,['balance','balance_push','weight_shift'],[])
    assert values['01_总览/主训宏平均成功率']==.5
    assert values['01_总览/主训宏平均稳定率']==pytest.approx(.6)
    assert values['01_总览/复习宏平均成功率'] is None


def test_legacy_checkpoint_does_not_invent_fall_or_goal_rates():
    c=Curriculum('balance')
    c.record(dict(stage='balance',stable=1,success=1,duration=10,team=0))
    assert c.metrics()['fall_rate'] is None
    assert c.metrics()['red_goal_rate'] is None
    c.monitor.clear()
    c.record(dict(stage='balance',stable=0,success=0,duration=10,team=0,fallen=1,goal=-1))
    assert c.metrics()['fall_rate']==1
    assert c.metrics()['blue_goal_rate']==1
