"""Create a compact workspace; importing this module never writes to W&B."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from q1_4v4.course_config import COURSES
from q1_4v4.telemetry import dashboard_specs
from q1_4v4.course_table import COURSE_TABLE_KEY


def main():
    import wandb
    import wandb_workspaces.workspaces as ws
    from wandb_workspaces.reports.v2 import LinePlot,MediaBrowser,Layout
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    meta=json.loads((root/'logs/active_training.json').read_text())
    run_id=args.run_id or meta['wandb_run_id']
    run=wandb.Api().run('mohismlab-robotics/q1-humanoid-kick-4v4/'+run_id)
    parallel=run.config.get('curriculum_mode','serial')=='parallel'
    per_course=run.config.get('per_course_metrics',False)
    # Legacy serial keys span all historical stages: do not mislabel them as one course.
    names=COURSES if per_course else []
    sections=[]
    for i,(title,panels) in enumerate(dashboard_specs(names,parallel,per_course)):
        sections.append(ws.Section(name=title,is_open=i==0,pinned=i==0,panels=[
            LinePlot(title=p['title'],x='iteration',y=p['keys'],
                     smoothing_factor=.2,smoothing_type='exponential') for p in panels]))
    if per_course:
        sections.insert(0,ws.Section(name='00 实时课程表 · 随训练更新',is_open=True,pinned=True,
            panels=[MediaBrowser(title='实时课程表（最新轮次）',media_keys=[COURSE_TABLE_KEY],num_columns=1,mode='gallery',gallery_axis='run',layout=Layout(w=24,h=18))]))
    if not per_course:
        sections[0].panels[2].title='03 红蓝平均单步奖励（旧采集口径含全部机器人）'
    view=ws.Workspace(entity='mohismlab-robotics',project='q1-humanoid-kick-4v4',
        name='Q1 4v4 六图总览',auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(query=run.name),sections=sections)
    existing=root/'artifacts/dashboard.json'
    if existing.exists():
        old=ws.Workspace.from_url(json.loads(existing.read_text())['url'].replace('/wandb/','/',1))
        old.sections=view.sections;old.runset_settings=view.runset_settings
        old._auto_generate_panels=False
        view=old
    view.save()
    url=run.url.split('/runs/',1)[0]+'?'+view.url.split('?',1)[1]
    existing.parent.mkdir(parents=True,exist_ok=True)
    existing.write_text(json.dumps({'url':url,'run_url':run.url,'run_id':run_id},indent=2))
    print(json.dumps({'url':url,'sections':[(s.name,len(s.panels)) for s in view.sections]},ensure_ascii=False))


if __name__=='__main__':main()
