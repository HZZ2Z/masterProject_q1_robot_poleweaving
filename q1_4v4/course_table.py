"""Curriculum quality and global field allocations as standard W&B HTML media."""
from html import escape
from math import isfinite
from .course_config import COURSES,SUCCESS
from .parallel_curriculum import PREREQUISITES

COURSE_TABLE_KEY='00_课程表/实时课程状态'
COLUMNS=('课程','足球场数','占比','当前成功率','稳定率','状态')
LABELS=('站立','扰动站立','重心转移','原地踏步','慢走','前进','后退','侧移',
        '速度跟踪','转向','行走转向','停步','启停','接近球','对齐','左脚正踢','右脚正踢',
        '左脚侧踢','右脚侧踢','踢后恢复','盘带','传接球','守门','1v1','2v1','2v2','团队配合','4v4比赛')


def percent(value):
    if value is None or not isinstance(value,(int,float)) or not isfinite(value):return '—'
    return f'{value*100:.2f}%'


def course_rows(summary,parallel=True):
    def metric(name,key):return summary.get('06_课程/'+name+'/'+key)
    rows=[];current=summary.get('01_总览/课程编号')
    direct_start=summary.get('01_总览/直接起始课程编号')
    counts={name:int(metric(name,'球场数') or 0) if parallel or current==i else 0
            for i,name in enumerate(COURSES)}
    total=summary.get('04_性能/并行球场总数') or sum(counts.values())
    for i,name in enumerate(COURSES):
        success=metric(name,'成功率');stable=metric(name,'稳定率')
        if parallel:
            values=[metric(name,k) for k in ('红队成功率','蓝队成功率','红队稳定率','蓝队稳定率')]
            known=all(v is not None and isinstance(v,(int,float)) and isfinite(v) for v in values)
            passed=known and all(v>=t for v,t in zip(values,(SUCCESS[name],SUCCESS[name],.9,.9)))
            if metric(name,'已掌握')==1:status='复习·待采样' if not known else '复习·达标' if passed else '复习·退化'
            elif name=='match' and counts[name]>0:status='终局训练'
            elif all(metric(parent,'已掌握')==1 for parent in PREREQUISITES[name]):status='主训·达标待认证' if passed else '主训'
            elif (metric(name,'球场数') or 0)>0:status='预习'
            else:status='未解锁'
        else:
            status='主训' if current==i else '已通过·无复测' if current is not None and i<current else '待训练'
            if current!=i and direct_start is not None and i<direct_start:status='本次跳过'
            if current!=i:success=stable=None
        worlds=counts[name]
        rows.append(dict(zip(COLUMNS,(f'{i} {LABELS[i]}',str(worlds),percent(worlds/total if total else 0),
                                      percent(success),percent(stable),status))))
    return rows


def course_table_html(summary,iteration,parallel=True):
    rows=course_rows(summary,parallel)
    total=int(summary.get('04_性能/并行球场总数') or sum(int(r['足球场数']) for r in rows))
    sampled=sum(int(r['足球场数'])>0 for r in rows)
    match_detail=''
    if 'debug/比赛步态奖励启用' in summary:
        mode='开启' if summary['debug/比赛步态奖励启用'] else '关闭'
        match_detail+=f'<p>比赛步态与动作风格奖励：{mode}。关闭时滑移、站距、关节速度、动作平滑及四项步态Reward均为0。</p>'
    for name in ('teamplay','match'):
        p='06_课程/'+name+'/'
        if p+'比赛子阶段' not in summary:continue
        reasons={'samples':'等待完整回合','posture':'整体姿态或复位未达标',
                 'owner_posture':'开球队员姿态未达标','owner_respawns':'开球队员复位过多',
                 'short_episode':'短回合时长未达标','owner_quality':'开球队员综合条件未达标','touch':'脚触球未达标',
                 'advance':'有效推进未达标','mid_touch':'中距脚触球未达标','mid_advance':'中距推进未达标',
                 'goals':'标准比赛进球未达标','pass':'累计通过窗口','expanded':'已扩大接近距离'}
        reason=reasons.get(summary.get(p+'子阶段门控'),'等待采样')
        match_detail+=f'<p>{escape(name)} 子阶段：{int(summary[p+"比赛子阶段"])}/4 · 近球训练场：{int(summary.get(p+"近球训练球场数",0)):,} · 中距场：{int(summary.get(p+"中距训练球场数",0)):,} · 远距场：{int(summary.get(p+"远距训练球场数",0)):,} · 标准阵型场：{int(summary.get(p+"标准阵型球场数",0)):,} · 近球距离上限：{summary.get(p+"近球距离上限_米",0):.2f}m · 门控：{reason} · 连续窗口：{int(summary.get(p+"子阶段连续通过窗口",0))}/3</p>'
    # Place currently sampled courses first, retaining each course number in its label.
    rows.sort(key=lambda r:(int(r['足球场数'])==0,r['状态'] in ('未解锁','待训练','本次跳过')))
    header=''.join('<th>'+escape(c)+'</th>' for c in COLUMNS)
    body=''.join('<tr class="'+('regression' if '退化' in r['状态'] else 'training' if '主训' in r['状态'] else '')+'">'+
                 ''.join('<td>'+escape(r[c])+'</td>' for c in COLUMNS)+'</tr>' for r in rows)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{color-scheme:light dark}}body{{font:14px system-ui,sans-serif;margin:12px}}
h3{{margin:0 0 6px}}p{{font-size:12px;opacity:.75;margin:6px 0 12px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 12px;border-bottom:1px solid #8884;text-align:left}}
th{{position:sticky;top:0;background:Canvas}}td:nth-child(n+2):nth-child(-n+5){{text-align:right}}
.regression{{background:#ffb02025}}.training{{background:#2196f320}}
</style></head><body><h3>实时课程表 · 第{int(iteration)}轮</h3>
<p>全局足球场：{total:,}个 · 正在采样：{sampled}门课程。场数为全部GPU合计，占比为本课程场数÷全局总场数。</p>
{match_detail}
<p>随每轮训练更新。占比仅指球场分配；比赛成功率为红蓝按队统计的有效进球回合率。整队稳定率保留零摔倒标准，开球难度使用独立的开球队员安全率与中距任务门控。</p>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></body></html>"""
