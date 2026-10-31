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
<p>随每轮训练更新。成功率和稳定率使用各课近期回合窗口；退化是复习提示，不自动改变认证。</p>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></body></html>"""
