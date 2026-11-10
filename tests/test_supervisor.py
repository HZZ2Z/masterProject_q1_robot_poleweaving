"""Exercise recovery with CPU checkpoints and disposable processes, without training."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from q1_4v4.supervisor import CheckpointStore, ProcessRegistry, ResourceGuard, TrainingSupervisor, atomic_json


@pytest.fixture
def job(tmp_path):
    project=tmp_path/'project';output=project/'logs/train';output.mkdir(parents=True)
    script=project/'train_distributed.py'
    script.write_text('''import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output')+1])
(output/'child.json').write_text(json.dumps(dict(argv=sys.argv[1:],run_id=os.environ.get('WANDB_RUN_ID'),resume=os.environ.get('WANDB_RESUME'))))
time.sleep(120)
''')
    optimizer=dict(state={0:dict(step=torch.tensor(40.),exp_avg=torch.ones(2))},param_groups=[dict(params=[0])])
    checkpoint=dict(backend='warp',iteration=12,parameter_max_difference=0.,
                    teams=[dict(actor={'weight':torch.ones(2)},critic={'weight':torch.ones(2)},
                                actor_optimizer=optimizer,critic_optimizer=optimizer)],
                    curriculum=dict(index=27),match_controls=dict(level=0))
    torch.save(checkpoint,output/'last.pt')
    command=[sys.executable,str(script),'--output',str(output),'--iterations','100',
             '--start-course','27','--curriculum-mode','serial']
    meta=dict(pid=99999999,worker_pids=[],output=str(output),console_log=str(output/'console.log'),
              world_size=1,gpu_uuids=['GPU-test'],command=command,target_iteration=100,
              parent_iteration=10,wandb_run_id='test-run',status='running')
    atomic_json(project/'logs/active_training.json',meta)
    supervisor=TrainingSupervisor(project,tmp_path)
    supervisor.guard.available=lambda _: (True,'ready')
    supervisor.guard.environment=lambda _: dict(os.environ,WANDB_RUN_ID='test-run',WANDB_RESUME='must')
    supervisor.guard.network=lambda _: None
    processes=[]

    def start():
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        processes.append(process)
        for _ in range(100):
            if (output/'child.json').exists():break
            time.sleep(.01)
        meta.update(pid=process.pid,worker_pids=[process.pid])
        atomic_json(supervisor.metadata,meta)
        return process

    yield supervisor,meta,checkpoint,start,output
    if supervisor.child:processes.append(supervisor.child)
    for process in processes:
        if process.poll() is None:process.kill()
        process.wait(timeout=5)


def test_adopts_running_job_without_restarting(job):
    supervisor,meta,_,start,_=job;process=start()
    supervisor.tick()
    assert process.poll() is None
    assert supervisor.state['status']=='watching'
    assert supervisor.state['restarts']==0
    assert supervisor.state['verified_iteration']==12
    assert supervisor.state['alive_workers']==1
    assert (supervisor.directory/'last_good.pt').exists()


def test_recovers_exited_process_with_same_run_and_remaining_iterations(job):
    supervisor,_,_,start,output=job;process=start();supervisor.tick()
    process.kill();process.wait();(output/'child.json').unlink()
    supervisor.tick()
    for _ in range(100):
        if (output/'child.json').exists():break
        time.sleep(.01)
    child=json.loads((output/'child.json').read_text());args=child['argv']
    assert supervisor.state['restarts']==1
    assert child['run_id']=='test-run' and child['resume']=='must'
    assert '--start-course' not in args
    assert args[args.index('--iterations')+1]=='88'
    assert args[args.index('--curriculum-mode')+1]=='serial'
    resumed=CheckpointStore.read(args[args.index('--checkpoint')+1])
    assert resumed['iteration']==12 and resumed['match_controls']==dict(level=0)
    torch.testing.assert_close(resumed['teams'][0]['actor_optimizer']['state'][0]['exp_avg'],torch.ones(2))


def test_backup_fallback_rejects_corrupt_and_nonfinite_checkpoints(job):
    supervisor,_,checkpoint,_,output=job
    store=CheckpointStore(output,supervisor.directory);store.refresh()
    (output/'last.pt').write_bytes(b'\x80broken checkpoint')
    _,resumed=store.resume();assert resumed['iteration']==12
    checkpoint['teams'][0]['actor_optimizer']['state'][0]['exp_avg'][0]=float('nan')
    torch.save(checkpoint,output/'last.pt')
    with pytest.raises(ValueError,match='Nonfinite optimizer'):store.refresh()
    assert store.resume()[1]['iteration']==12


def test_backup_cannot_be_used_for_another_output(job,tmp_path):
    supervisor,_,_,_,output=job;store=CheckpointStore(output,supervisor.directory);store.refresh()
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(RuntimeError,match='No valid checkpoint'):
        CheckpointStore(other,supervisor.directory).resume()


def test_gpu_busy_defers_recovery_without_launching(job):
    supervisor,_,_,_,_=job
    supervisor.guard.available=lambda _: (False,'GPU occupied by another task')
    supervisor.tick()
    assert supervisor.state['status']=='waiting_gpu'
    assert supervisor.child is None and supervisor.state['restarts']==0


def test_network_failure_defers_recovery(job):
    supervisor,_,_,_,_=job
    def disconnected(_):raise TimeoutError('network unavailable')
    supervisor.guard.network=disconnected;supervisor.tick()
    assert supervisor.state['status']=='restart_wait'
    assert supervisor.child is None and supervisor.state['restarts']==0


@pytest.mark.parametrize('status',['stopped','paused','completed','stopping_for_maintenance'])
def test_manual_stop_prevents_recovery(job,status):
    supervisor,meta,_,_,_=job;meta['status']=status;atomic_json(supervisor.metadata,meta)
    supervisor.tick()
    assert supervisor.state['status']=='manual_stop' and supervisor.child is None


def test_startup_accepts_old_checkpoint_mtime(job):
    supervisor,meta,_,start,output=job;process=start()
    meta.update(status='starting',parent_iteration=12);atomic_json(supervisor.metadata,meta)
    old=time.time()-3600;os.utime(output/'last.pt',(old,old))
    supervisor.tick()
    assert process.poll() is None and supervisor.state['status']=='watching'
    assert supervisor.shutdown_at is None


def test_stalled_training_signals_only_verified_process(job,monkeypatch):
    supervisor,_,_,start,output=job;process=start();supervisor.tick()
    old=time.time()-700;os.utime(output/'last.pt',(old,old))
    signals=[];monkeypatch.setattr(supervisor.registry,'signal',lambda identity,number:signals.append((identity,number)))
    supervisor.tick()
    assert signals==[(supervisor.registry.inspect(process.pid),signal.SIGTERM)]
    assert supervisor.state['status']=='stopping_failed_training'


def test_process_identity_blocks_pid_reuse_and_unrelated_process(job,monkeypatch):
    supervisor,_,_,start,output=job;process=start();registry=ProcessRegistry(output)
    identity=registry.inspect(process.pid);assert identity
    assert registry.inspect(os.getpid()) is None
    calls=[];monkeypatch.setattr(os,'kill',lambda *args:calls.append(args))
    registry.signal(dict(identity,start_ticks='incorrect'),signal.SIGTERM)
    assert calls==[]


def test_supervisor_rejects_other_workspace_and_replacement_run(job,tmp_path):
    supervisor,meta,_,_,_=job
    supervisor.metadata_read();meta['wandb_run_id']='another-run';atomic_json(supervisor.metadata,meta)
    supervisor.tick()
    assert supervisor.stop_event.is_set() and supervisor.child is None
    with pytest.raises(ValueError,match='outside authorized'):
        TrainingSupervisor(tmp_path/'outside',tmp_path/'allowed')


def test_gpu_gate_checks_memory_utilization_and_compute_processes(tmp_path):
    guard=ResourceGuard(tmp_path);meta=dict(gpu_uuids=['GPU-allowed'])
    def query(arguments):
        return 'GPU-allowed, 0, 0\nGPU-other, 90000, 100\n' if arguments[0].startswith('--query-gpu=') else ''
    guard.query=query;assert guard.available(meta)==(True,'ready')
    guard.query=lambda args: 'GPU-allowed, 10, 100\n' if args[0].startswith('--query-gpu=') else ''
    assert not guard.available(meta)[0]
    guard.query=lambda args: 'GPU-allowed, 0, 0\n' if args[0].startswith('--query-gpu=') else '123, GPU-allowed\n'
    assert not guard.available(meta)[0]


def test_completed_target_and_stop_marker_do_not_launch(job):
    supervisor,meta,_,_,_=job;meta['target_iteration']=12;atomic_json(supervisor.metadata,meta)
    supervisor.tick()
    assert supervisor.state['status']=='completed' and supervisor.child is None
    supervisor.stop_event.clear();(supervisor.directory/'STOP').touch();supervisor.tick()
    assert supervisor.stop_event.is_set() and supervisor.child is None


def test_detached_cli_start_status_and_stop_leave_training_running(job):
    supervisor,_,_,start,_=job;process=start()
    script=Path(__file__).resolve().parents[1]/'courses/soccer-4v4/scripts/watch_training.py'
    def cli(action):
        result=subprocess.check_output([sys.executable,str(script),action,'--project',str(supervisor.project),
                                        '--root',str(supervisor.root),'--interval','1'],text=True)
        return json.loads(result)
    try:
        launched=cli('start');assert launched['started']
        for _ in range(100):
            state=cli('status')
            if state.get('status')=='watching':break
            time.sleep(.02)
        assert state['supervisor_alive'] and state['status']=='watching'
        assert cli('start')['already_running']
        assert process.poll() is None
    finally:
        cli('stop')
        for _ in range(100):
            if not (supervisor.directory/'monitor.pid').exists():break
            time.sleep(.02)
    assert process.poll() is None
    assert not cli('status')['supervisor_alive']
