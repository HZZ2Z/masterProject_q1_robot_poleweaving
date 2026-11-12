"""Explicit per-course metric units and six-chart workspace definitions."""
from .course_config import COURSES,GAIT_COURSES,KICKS,DURATIONS,SUCCESS
from .match_task import MatchTask

SKILL_NAMES=('upright','height','forward','side','left','right','task_distance')
MATCH_TERM_LABELS=('存活','直立','基座高度','关节速度','动作平滑','支撑滑移','非脚部着地',
                   '追球','角色移动','角色站位','合法控球Potential','首次脚触球','有效进球','实际失球','机器人复位','球出界','非脚部触球',
                   '站距','追球距离','首次推进','移动速度跟踪','有效摆腿','有效落脚交替','支撑异常',
                   '脚部接近','近球停滞','远球停滞','有效正踢','有效侧踢','向门新增净推进','动作示范AMP','非守门员进入禁区',
                   '支撑脚滑移约束','姿态边界约束','持续腾空约束','上肢摆动约束')
assert len(MATCH_TERM_LABELS)==len(MatchTask.terms)


def match_metrics(name,values,terms,amplitude,control=None,standard_worlds=0,near_worlds=0,bucket_worlds=None):
    prefix='06_课程/'+name+'/'
    data={prefix+'红队站立时间占比':values['standing_red'],prefix+'蓝队站立时间占比':values['standing_blue'],
          prefix+'红队每机器人分钟复位次数':values['respawns_red_per_min'],
          prefix+'蓝队每机器人分钟复位次数':values['respawns_blue_per_min'],
          prefix+'红队每机器人分钟摔倒次数':values['falls_red_per_min'],
          prefix+'蓝队每机器人分钟摔倒次数':values['falls_blue_per_min'],
          prefix+'红队追球者球距_米':values['chaser_distance_red'],
          prefix+'蓝队追球者球距_米':values['chaser_distance_blue'],
          prefix+'控制动作幅度':amplitude,'debug/'+name+'_reward_sum_residual':values['reward_residual']}
    for team,label in (('red','红队'),('blue','蓝队')):
        for key,title in (('foot_gap','追球者脚球间距_米'),('near_fraction','追球者近球时间占比'),
                          ('near_stall','追球者近球停滞时间占比'),('far_stall','追球者远球停滞时间占比'),
                          ('left_strike','左脚有效踢球控制步占比'),('right_strike','右脚有效踢球控制步占比')):
            data[prefix+label+title]=values.get(key+'_'+team,0.)
        moving=values.get('walk_fraction_'+team,0.)
        data[prefix+label+'移动指令占比']=moving
        data[prefix+label+'移动时速度误差_mps']=values.get('walk_error_'+team) if moving else None
        data[prefix+label+'移动时摆腿得分']=values.get('walk_lift_'+team) if moving else None
    if control is not None:
        data.update({prefix+'比赛子阶段':control.level,prefix+'近球脚球间距上限_米':.25,
                     prefix+'子阶段连续通过窗口':control.windows,prefix+'子阶段门控':control.reason,
                     prefix+'近球训练球场数':near_worlds,prefix+'标准阵型球场数':standard_worlds,
                     prefix+'标准阵型占比':standard_worlds/max(sum((bucket_worlds or {}).values()) or standard_worlds+near_worlds,1)})
        if bucket_worlds:
            for key,title in (('mid','中距训练球场数'),('far','远距训练球场数')):data[prefix+title]=bucket_worlds[key]
        rates=control.rates()
        for team,label in (('red','红队'),('blue','蓝队')):
            for key,title in (('touch','近球开球队脚触球率'),('advance','近球开球队有效推进率'),
                              ('quality','近球开球队姿态稳定率'),('standard_touch','标准阵型脚触球率'),
                              ('standard_goal','标准阵型有效进球率'),('kick','近球开球队有效踢球率'),
                              ('mid_kick','中距开球队有效踢球率')):
                data[prefix+label+title]=rates[key+'_'+team]
            for key,title in (('training_quality','开球队员安全通过率'),('mid_touch','中距开球队脚触球率'),
                              ('far_touch','远距开球队脚触球率'),('mid_advance','中距开球队有效推进率'),('far_advance','远距开球队有效推进率')):
                data[prefix+label+title]=rates[key+'_'+team]
            for key,title in (('owner_pose_pass','开球队员姿态通过率'),('owner_reset_pass','开球队员复位通过率'),
                ('owner_time_pass','开球队员时长通过率'),('owner_healthy_sum','开球队员健康姿态时间占比'),
                ('owner_bad_up','开球队员倾斜时间占比'),('owner_bad_height','开球队员高度异常时间占比'),
                ('owner_bad_width','开球队员站距异常时间占比'),('owner_bad_contact','开球队员非脚部着地时间占比')):
                data[prefix+label+title]=rates[key+'_'+team]
    for side,label in (('red','红队'),('blue','蓝队')):
        for key,title in (('slip','约束接触滑移_mps'),('flight','移动腾空时间占比'),
                          ('quality','移动质量系数'),('upper_error','上肢偏离中立_弧度')):
            data[prefix+label+title]=values.get('motion_'+key+'_'+side,0.)
    for label,pair in zip(MATCH_TERM_LABELS,terms):
        for team,value in zip(('红队','蓝队'),pair):data[f'07_Reward/{name}/{team}/{label}']=value
    return data


def match_reward_specs(name):
    groups=[('01 基础姿态',('存活','直立','基座高度','站距')),
            ('02 动作与支撑约束',('关节速度','动作平滑','支撑滑移','非脚部着地','移动速度跟踪','有效摆腿','有效落脚交替','支撑异常',
                                '支撑脚滑移约束','姿态边界约束','持续腾空约束','上肢摆动约束')),
            ('03 追球与角色',('追球','追球距离','角色移动','角色站位','脚部接近')),
            ('04 有效踢球与停滞',('近球停滞','远球停滞','有效正踢','有效侧踢','向门新增净推进','动作示范AMP')),
            ('05 合法控球与首次触球推进',('合法控球Potential','首次脚触球','首次推进')),
            ('06 进球与规则事件',('有效进球','实际失球','机器人复位','球出界','非脚部触球','非守门员进入禁区'))]
    return (f'{COURSES.index(name):02d} Reward分解 · {name}',[
        {'title':title+'（平均每控制步）','keys':[f'07_Reward/{name}/{team}/{label}' for label in labels for team in ('红队','蓝队')]}
        for title,labels in groups])


def active_team_reward(reward,active):
    values=reward.squeeze(-1).reshape(-1,2,4)
    mask=active.reshape(-1,2,4)
    return (values*mask).sum((0,2))/mask.sum((0,2)).clamp(min=1)


def course_metrics(name,stats,reward,gait,skill,worlds,total_worlds,mastered=False):
    prefix='06_课程/'+name+'/'
    known=stats['red_samples']+stats['blue_samples']>0
    data={'红队平均单步奖励':reward[0],'蓝队平均单步奖励':reward[1],
          '成功率':stats['success_rate'] if known else None,'稳定率':stats['stable_rate'] if known else None,
          '红队成功率':stats['red_success'] if stats['red_samples'] else None,'蓝队成功率':stats['blue_success'] if stats['blue_samples'] else None,
          '红队稳定率':stats['red_stability'] if stats['red_samples'] else None,'蓝队稳定率':stats['blue_stability'] if stats['blue_samples'] else None,
          '摔倒率':stats.get('fall_rate'),'平均回合秒数':stats['episode_seconds'] if known else None,
          '回合上限秒数':DURATIONS[name]*.02,'球场数':worlds,'球场占比':worlds/max(total_worlds,1),
          '红队窗口样本':stats['red_gate_samples'],'蓝队窗口样本':stats['blue_gate_samples'],
          '阶段更新次数':stats['stage_updates'],'连续通过窗口':stats['passed_windows'],
          '窗口认证进度':min(stats['passed_windows']/3,1),
          '样本资格':min(min(stats['red_gate_samples'],stats['blue_gate_samples'])/256,1),
          '更新资格':min(stats['stage_updates']/32,1),'已掌握':int(mastered),
          '目标成功率':SUCCESS[name],'目标稳定率':.9,
          '最大已达净推进_米':stats['progress_m'] if known else None,
          '脚部触球率':stats['foot_touch_rate'] if known else None,
          '触球后恢复率':stats['recovery_rate'] if known else None,
          '红队进球比例':stats.get('red_goal_rate'),'蓝队进球比例':stats.get('blue_goal_rate')}
    if worlds and skill is not None:
        data.update({'基座平均高度_米':skill['height'],'平均球距_米':skill['task_distance']})
    if worlds and gait is not None:
        if name=='step_in_place':
            for key,label in [('knee_lift','抬膝得分'),('position_drift','偏离起点_米'),
                              ('heading_error','航向偏差_弧度'),('stance_width','双脚横向间距_米'),
                              ('single_support','单支撑期正确支撑率'),('knee_signal_rate','抬膝有效奖励比例')]:
                data[label]=gait.get(key)
            data['准备奖励_单步']=gait.get('marching_progress',0.)*.02
        for key,label in [('left_steps','左脚过程累计步数'),('right_steps','右脚过程累计步数'),
                          ('alternations','过程累计交替次数'),('weight_shifts','过程累计重心切换'),
                          ('slip_mps','支撑滑移_mps'),('phase_match','相位匹配率'),('phase_lift','抬脚得分')]:
            data[label]=gait[key] if key not in ('phase_match','phase_lift') or name in GAIT_COURSES else None
    return {prefix+k:v for k,v in data.items()}


def macro_metrics(course_stats,ready,mastered):
    def mean(names,key):
        values=[course_stats[n][key] for n in names if n in course_stats
                and course_stats[n]['red_samples']+course_stats[n]['blue_samples']>0]
        return sum(values)/len(values) if values else None
    return {'01_总览/主训宏平均成功率':mean(ready,'success_rate'),
            '01_总览/主训宏平均稳定率':mean(ready,'stable_rate'),
            '01_总览/复习宏平均成功率':mean(mastered,'success_rate')}


def dashboard_specs(names,parallel=True,per_course=True):
    """Pure specification: every section is capped at six single-unit panels."""
    def panel(title,*keys):return {'title':title,'keys':list(keys)}
    if parallel:
        coverage=panel('01 课程覆盖（数量）','01_总览/已掌握课程数','01_总览/已解锁课程数','01_总览/主训课程数')
        quality=panel('02 主训质量（各课程等权宏平均）','01_总览/主训宏平均成功率','01_总览/主训宏平均稳定率','01_总览/复习宏平均成功率')
    else:
        coverage=panel('01 当前串行课程编号','01_总览/课程编号')
        quality=panel('02 当前课程成功率与稳定率','01_总览/阶段成功率','01_总览/稳定率')
    sections=[('00 整体训练：六图总览',[
        coverage,quality,panel('03 红蓝有效参与者平均单步奖励','01_总览/红队回报','01_总览/蓝队回报'),
        panel('04 共享PPO更新幅度：近似KL','debug/shared_approx_kl'),
        panel('05 全局球场控制步/秒','04_性能/球场步每秒'),
        panel('06 各卡训练进程显存（GiB）',*[f'04_性能/GPU{i}显存_GiB' for i in range(5)])])]
    for name in COURSES:
        if name not in names:continue
        p='06_课程/'+name+'/'
        if not per_course:
            # Existing serial jobs can use the compact view without a restart.
            charts=[panel('01 红蓝平均单步奖励','01_总览/红队回报','01_总览/蓝队回报'),
                    panel('02 红蓝任务成功率','02_晋级/红队成功率','02_晋级/蓝队成功率'),
                    panel('03 红蓝稳定率','02_晋级/红队稳定率','02_晋级/蓝队稳定率'),
                    panel('04 连续通过窗口（达到3晋级）','02_晋级/连续通过窗口'),
                    panel('05 平均回合时长（秒）','01_总览/回合时长_秒'),
                    panel('06 基座平均高度（米）','01_总览/平均身高_米')]
        else:
            charts=[panel('01 红蓝平均单步奖励',p+'红队平均单步奖励',p+'蓝队平均单步奖励'),
                    panel(f'02 红蓝任务成功率（目标{SUCCESS[name]:.0%}）',p+'红队成功率',p+'蓝队成功率',p+'目标成功率'),
                    panel('03 稳定率与摔倒率（近期窗口）',p+'红队稳定率',p+'蓝队稳定率',p+'摔倒率',p+'目标稳定率'),
                    panel('04 配额与认证进度（0—1；资格1表示达标）',p+'球场占比',p+'窗口认证进度',p+'样本资格',p+'更新资格',p+'已掌握')]
            if name=='match':
                charts[1]=panel('02 近球触球与中近距有效踢球回合率',*[p+t+k for t in ('红队','蓝队') for k in ('近球开球队脚触球率','近球开球队有效踢球率','中距开球队有效踢球率')])
                charts[3]=panel('04 有效进球比例：全部与标准阵型',p+'红队成功率',p+'蓝队成功率',p+'红队标准阵型有效进球率',p+'蓝队标准阵型有效进球率')
                charts[2]=panel('03 健康姿态时间占比与整队姿态稳定率',p+'红队站立时间占比',p+'蓝队站立时间占比',p+'红队稳定率',p+'蓝队稳定率')
                charts += [panel('05 追球者躯干球距与脚球间距（米）',*[p+t+k for t in ('红队','蓝队') for k in ('追球者球距_米','追球者脚球间距_米')]),
                           panel('06 每机器人分钟复位次数',p+'红队每机器人分钟复位次数',p+'蓝队每机器人分钟复位次数')]
            elif name in ('balance','balance_push'):
                charts += [panel('05 平均回合时长（秒）',p+'平均回合秒数',p+'回合上限秒数'),panel('06 基座平均高度（米）',p+'基座平均高度_米')]
            elif name=='weight_shift':
                charts += [panel('05 重心切换次数（过程累计均值）',p+'过程累计重心切换'),panel('06 支撑脚滑移（m/s）',p+'支撑滑移_mps')]
            elif name=='step_in_place':
                charts[0]=panel('01 红蓝Reward与准备奖励（每控制步）',p+'红队平均单步奖励',p+'蓝队平均单步奖励',p+'准备奖励_单步')
                charts += [panel('05 Phase、抬脚、抬膝与有效支撑（0—1）',p+'相位匹配率',p+'抬脚得分',p+'抬膝得分',p+'单支撑期正确支撑率',p+'抬膝有效奖励比例'),
                           panel('06 偏离起点与双脚间距（米）',p+'偏离起点_米',p+'双脚横向间距_米')]
            elif name in GAIT_COURSES and name!='approach':
                charts += [panel('05 接触相位匹配与抬脚得分（0—1）',p+'相位匹配率',p+'抬脚得分'),
                           panel('06 左右落脚和交替（过程累计均值，非终局次数）',p+'左脚过程累计步数',p+'右脚过程累计步数',p+'过程累计交替次数')]
            elif name in ('approach','align','pass_receive'):
                charts += [panel('05 平均球距（米；传接球为接球人球距）',p+'平均球距_米'),panel('06 脚部触球回合比例',p+'脚部触球率')]
            elif name in KICKS or name in ('dribble','teamplay'):
                charts += [panel('05 回合内最大已达净推进（米）',p+'最大已达净推进_米'),panel('06 脚触球与触球后恢复比例',p+'脚部触球率',p+'触球后恢复率')]
            else:
                charts += [panel('05 平均回合时长（秒）',p+'平均回合秒数',p+'回合上限秒数'),panel('06 回合结束时进球方向比例（非胜率）',p+'红队进球比例',p+'蓝队进球比例')]
        sections.append((f'{COURSES.index(name):02d} 课程 · {name}',charts))
        if per_course and name in ('teamplay','match'):
            sections.append(match_reward_specs(name))
            sections.append((f'{COURSES.index(name):02d} 踢球与场景 · {name}',[
                panel('01 追球者近球与远球停滞时间占比',*[p+t+k for t in ('红队','蓝队') for k in ('追球者近球停滞时间占比','追球者远球停滞时间占比')]),
                panel('02 左右脚有效踢球控制步占比',*[p+t+k for t in ('红队','蓝队') for k in ('左脚有效踢球控制步占比','右脚有效踢球控制步占比')]),
                panel('03 中远距与标准阵型脚触球率',*[p+t+k for t in ('红队','蓝队') for k in ('中距开球队脚触球率','远距开球队脚触球率','标准阵型脚触球率')]),
                panel('04 各类开球场数',*[p+k for k in ('近球训练球场数','中距训练球场数','远距训练球场数','标准阵型球场数')]),
                panel('05 中近距开球队有效踢球率',*[p+t+k for t in ('红队','蓝队') for k in ('近球开球队有效踢球率','中距开球队有效踢球率')]),
                panel('06 开球队员安全率与整队稳定率',*[p+t+k for t in ('红队','蓝队') for k in ('开球队员安全通过率','近球开球队姿态稳定率')])]))
            sections.append((f'{COURSES.index(name):02d} 支撑与姿态 · {name}',[
                panel('01 开球队员姿态、复位、时长与综合通过率',*[p+t+k for t in ('红队','蓝队') for k in ('开球队员姿态通过率','开球队员复位通过率','开球队员时长通过率','开球队员安全通过率')]),
                panel('02 开球队员姿态异常时间占比',*[p+t+k for t in ('红队','蓝队') for k in ('开球队员倾斜时间占比','开球队员高度异常时间占比','开球队员站距异常时间占比','开球队员非脚部着地时间占比')]),
                panel('03 支撑接触滑移（m/s）',p+'红队约束接触滑移_mps',p+'蓝队约束接触滑移_mps'),
                panel('04 AMP示范奖励与接触姿态约束（每控制步）',*[f'07_Reward/{name}/{t}/{k}' for t in ('红队','蓝队') for k in ('动作示范AMP','支撑脚滑移约束','姿态边界约束','持续腾空约束','上肢摆动约束')]),
                panel('05 移动质量与腾空比例',*[p+t+k for t in ('红队','蓝队') for k in ('移动质量系数','移动腾空时间占比')]),
                panel('06 上肢偏离中立（弧度）',p+'红队上肢偏离中立_弧度',p+'蓝队上肢偏离中立_弧度')]))
    if per_course:sections.append(('98 动作示范与历史编码',[
        panel('01 足球与AMP加权优势绝对值','debug/shared_task_advantage_abs','debug/shared_amp_advantage_abs'),
        panel('02 专家与有效运动策略评分','debug/amp_expert_score','debug/amp_effective_policy_score'),
        panel('03 足球与AMP价值损失','debug/shared_task_value_loss','debug/shared_amp_value_loss'),
        panel('04 AMP发放资格占比','debug/amp_eligible_fraction'),
        panel('05 有效运动风格得分','debug/amp_effective_style'),
        panel('06 历史编码输出权重范数','debug/history_output_weight_norm')]))
    sections.append(('99 学习诊断（按需展开）',[
        panel('价值损失','debug/shared_value_loss'),panel('策略损失','debug/shared_policy_loss'),
        panel('连续动作分布总熵','debug/shared_dist_entropy'),panel('PPO裁剪比例','debug/shared_clip_fraction'),
        panel('梯度范数（裁剪前）','debug/shared_actor_grad_norm','debug/shared_critic_grad_norm'),
        panel('最大训练进程显存（GiB）','04_性能/进程显存_GiB')]))
    assert all(len(charts)==6 for _,charts in sections)
    return sections
