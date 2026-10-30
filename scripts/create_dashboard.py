"""Create an intentional W&B view rather than an automatically generated chart wall."""
import json
from pathlib import Path
import wandb_workspaces.workspaces as ws
from wandb_workspaces.reports.v2 import LinePlot

root=Path(__file__).resolve().parents[1]
meta=json.loads((root/'logs/active_training.json').read_text())
run=json.loads((Path(meta['output'])/'wandb_run.json').read_text())

def chart(title,*metrics):
    return LinePlot(title=title,x='iteration',y=list(metrics),smoothing_factor=.2,smoothing_type='exponential')

view=ws.Workspace(entity='mohismlab-robotics',project='q1-humanoid-kick-4v4',name='Q1 4v4 训练总览',
    auto_generate_panels=False,runset_settings=ws.RunsetSettings(query=meta.get('run_name',run['id'])),sections=[
    ws.Section(name='01 总览',is_open=True,pinned=True,panels=[
        chart('课程进度','01_总览/课程编号'),
        chart('成功率与稳定率','01_总览/阶段成功率','01_总览/稳定率','01_总览/回合摔倒率'),
        chart('红蓝回报','01_总览/红队回报','01_总览/蓝队回报'),
        chart('回合存活秒数','01_总览/回合时长_秒'),
        chart('身体高度','01_总览/平均身高_米')]),
    ws.Section(name='02 晋级门槛',is_open=True,panels=[
        chart('连续通过窗口：达到3才晋级','02_晋级/连续通过窗口'),
        chart('当前窗口完成回合','02_晋级/窗口样本数'),
        chart('红蓝窗口样本：各需256','02_晋级/红队窗口样本','02_晋级/蓝队窗口样本'),
        chart('阶段更新次数','02_晋级/阶段更新次数'),
        chart('红蓝分别过关','02_晋级/红队成功率','02_晋级/蓝队成功率'),
        chart('红蓝分别站稳：各需85%','02_晋级/红队稳定率','02_晋级/蓝队稳定率')]),
    ws.Section(name='03 足球技能',is_open=True,panels=[
        chart('左右脚触球','03_技能/左脚触球','03_技能/右脚触球'),
        chart('正踢与侧踢','03_技能/正踢奖励','03_技能/侧踢奖励'),
        chart('有效触球与踢后恢复','03_技能/脚部触球率','03_技能/触球后恢复率'),
        chart('净推进距离','03_技能/净推进_米')]),
    ws.Section(name='05 步态质量',is_open=True,panels=[
        chart('支撑脚滑移：目标≤0.05m/s','05_步态/支撑脚滑移_mps'),
        chart('接触相位匹配：目标≥80%','05_步态/接触相位匹配率'),
        chart('左右有效落脚','05_步态/左脚有效步数','05_步态/右脚有效步数'),
        chart('真实左右交替','05_步态/有效交替落脚'),
        chart('正确支撑与抬脚','05_步态/相位抬脚得分'),
        chart('重心切换次数','05_步态/有效重心切换'),
        chart('速度跟踪误差','05_步态/速度误差_mps'),
        chart('足底离地高度','05_步态/足底离地_米'),
        chart('非脚部接地','05_步态/非脚部着地比例')]),
    ws.Section(name='04 性能',is_open=False,panels=[
        chart('球场控制步/秒','04_性能/球场步每秒'),
        chart('GPU数量与并行球场','04_性能/GPU数量','04_性能/并行球场总数'),
        chart('各卡 Warp + Torch 显存','04_性能/GPU0显存_GiB','04_性能/GPU1显存_GiB','04_性能/GPU2显存_GiB','04_性能/GPU3显存_GiB','04_性能/GPU4显存_GiB'),
        chart('总显存','04_性能/GPU总显存_GiB'),
        chart('共享PPO近似KL','debug/shared_approx_kl'),
        chart('共享PPO裁剪比例','debug/shared_clip_fraction'),
        chart('共享策略熵','debug/shared_dist_entropy'),
        chart('共享价值损失','debug/shared_value_loss')])])
existing_path=root/'artifacts/dashboard.json'
if existing_path.exists():
    old_url=json.loads(existing_path.read_text())['url'].replace('/wandb/','/',1)
    old=ws.Workspace.from_url(old_url)
    old.sections=view.sections
    old.runset_settings=view.runset_settings
    view=old
view.save()
view_url=run['url'].split('/runs/',1)[0]+'?'+view.url.split('?',1)[1]
(root/'artifacts/dashboard.json').write_text(json.dumps({'url':view_url,'run_url':run['url'],'run_id':run['id']},indent=2))
print(view_url)
