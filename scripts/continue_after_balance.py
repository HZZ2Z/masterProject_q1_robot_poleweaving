"""Wait for genuine course-0 certification, then resume a staged parallel build.

All machine state, logs and deployment files must be below --root. Credentials
are supplied through WANDB_API_KEY in the process environment, never files.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request
import urllib.error
import torch


def balance_passed(checkpoint):
    state=checkpoint.get('curriculum',{})
    if state.get('kind')=='parallel':return 'balance' in state.get('mastered',[])
    return (state.get('version')==3 and state.get('stage') not in (None,'balance')
            and state.get('index',0)>=1 and checkpoint.get('policy_sharing')=='shared')


def running(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def worker_pids(manager,project):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if int((proc/'stat').read_text().rsplit(')',1)[1].split()[1])==manager and b'train_distributed.py' in (proc/'cmdline').read_bytes():
                if (proc/'cwd').resolve()!=project:raise RuntimeError('Worker outside authorized project')
                result.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result


class Transition:
    def __init__(self,root,candidate):
        self.root=root.resolve();self.project=self.root/'q1-humanoid-kick-4v4';self.candidate=candidate.resolve()
        assert self.candidate.is_relative_to(self.root)
        self.status_path=self.root/'artifacts/parallel_transition.json'
        self.python=os.environ['Q1_4V4_PYTHON'];self.log=self.root/'artifacts/parallel_transition.log'
        self.ready=json.loads((self.root/'artifacts/parallel_ready.json').read_text())
        self.manifest=json.loads((self.root/'artifacts/parallel_code_manifest.json').read_text())

    def status(self,state,**extra):
        payload=dict(state=state,watcher_pid=os.getpid(),time=time.time(),candidate_commit=self.ready['commit'],**extra)
        temporary=self.status_path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(self.status_path)
        print(json.dumps(payload),flush=True)

    def active(self):return json.loads((self.project/'logs/active_training.json').read_text())
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.project),*args],text=True).strip()

    def network_ok(self):
        try:
            urllib.request.urlopen(os.environ['WANDB_BASE_URL'],timeout=15).close()
        except urllib.error.HTTPError as error:
            if error.code>=500:raise
        return True

    def hardware(self,meta,allow_workers):
        probe=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
        devices={line.split(',')[0].strip():[int(x.strip()) for x in line.split(',')[1:]] for line in probe.splitlines()}
        for uuid in meta['gpu_uuids']:
            if uuid not in devices:raise RuntimeError('Requested GPU disappeared')
            if not allow_workers and (devices[uuid][0]>128 or devices[uuid][1]>=5):raise RuntimeError('GPU not idle')
        for line in apps.splitlines():
            pid,uuid=[x.strip() for x in line.split(',')]
            if uuid in meta['gpu_uuids'] and int(pid) not in allow_workers:raise RuntimeError('GPU occupied by another process')
        return devices

    def candidate_valid(self):
        assert self.ready['tests_passed']
        for name,digest in self.manifest.items():
            path=(self.candidate/name).resolve();assert path.is_relative_to(self.candidate)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        head=subprocess.check_output(['git','-c','safe.directory='+str(self.candidate),'-C',str(self.candidate),'rev-parse','HEAD'],text=True).strip()
        assert head==self.ready['commit']

    def stop(self,meta):
        workers=worker_pids(meta['pid'],self.project);assert len(workers)==meta['world_size']
        self.hardware(meta,set(workers));self.network_ok()
        assert not self.git('status','--porcelain'),'Live source has uncommitted changes'
        self.status('stopping_after_balance',old_manager_pid=meta['pid'],workers=workers)
        for pid in workers:os.kill(pid,signal.SIGTERM)
        deadline=time.time()+600
        while running(meta['pid']) and time.time()<deadline:time.sleep(5)
        if running(meta['pid']):raise RuntimeError('Trainer did not stop gracefully; no forced kill attempted')
        for _ in range(12):
            try:self.hardware(meta,set());return
            except RuntimeError:time.sleep(5)
        raise RuntimeError('GPUs did not become idle')

    def deploy(self):
        for name in self.manifest:
            target=self.project/name;assert target.resolve().is_relative_to(self.project)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.candidate/name,target)
        shutil.copytree(self.candidate/'.git',self.project/'.git',dirs_exist_ok=True)
        assert self.git('rev-parse','HEAD')==self.ready['commit']
        assert not self.git('status','--porcelain')

    def environment(self,meta):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(meta['gpu_uuids']),NCCL_SHM_DISABLE='1',
                   OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   WANDB_RUN_ID=meta['wandb_run_id'],WANDB_RESUME='must')
        return env

    def smoke(self,meta,source,checkpoint):
        output=self.project/'logs'/('parallel_transition_smoke_'+str(checkpoint['iteration']))
        command=[self.python,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(meta['world_size']),
                 'train_distributed.py','--curriculum-mode','parallel','--checkpoint',str(source),
                 '--num-envs','32','--horizon','8','--epochs','1','--iterations','2','--episode-steps','4',
                 '--nconmax','1536','--njmax','6144','--save-interval','1','--no-wandb','--job-type','smoke','--output',str(output)]
        self.status('testing_five_gpu_parallel_resume',source_iteration=checkpoint['iteration'])
        with (self.root/'artifacts/parallel_transition_smoke.log').open('w') as stream:
            result=subprocess.run(command,cwd=self.project,env=self.environment(meta),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError('Five-GPU parallel smoke failed')
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        assert saved['curriculum_mode']=='parallel' and balance_passed(saved)
        assert saved['world_size']==meta['world_size'] and saved['parameter_max_difference']==0
        assert len(saved['teams'])==1 and saved['iteration']==checkpoint['iteration']+2
        keys=[r['course_rng']['weight_shift'].numpy().tobytes() for r in saved['distributed_rng']]
        assert len(set(keys))==len(keys),'Duplicate per-rank curriculum sampling'

    def launch(self,meta,source,checkpoint,mode,worlds=None):
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        output=self.project/'logs'/(mode+'_'+stamp);output.mkdir()
        command=list(meta['command'])
        def option(flag,value):
            if flag in command:command[command.index(flag)+1]=str(value)
            else:command.extend((flag,str(value)))
        target=meta.get('target_iteration',meta.get('parent_iteration',0)+int(command[command.index('--iterations')+1]))
        assert target>checkpoint['iteration']
        option('--iterations',target-checkpoint['iteration']);option('--checkpoint',source)
        option('--curriculum-mode',mode);option('--output',output);option('--run-name',meta['run_name'])
        worlds=worlds or meta['num_envs'];option('--num-envs',worlds)
        with (output/'console.log').open('wb') as stream:
            child=subprocess.Popen(command,cwd=self.project,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   env=self.environment(meta),start_new_session=True)
        current=dict(meta,pid=child.pid,output=str(output),command=command,curriculum_mode=mode,
                     num_envs=worlds,total_num_envs=worlds*meta['world_size'],
                     course_name=checkpoint['curriculum']['stage'],start_course=None,
                     fresh_start=False,checkpoint_loaded=True,resumed_from=str(source),parent_iteration=checkpoint['iteration'],
                     previous_output=meta['output'],status='starting',git_commit=self.ready['commit'],
                     transition_reason='Course 0 passed; prerequisite-gated parallel curriculum',
                     created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        current.pop('worker_pids',None);current.pop('wandb_verified',None)
        (self.project/'logs/active_training.json').write_text(json.dumps(current,indent=2))
        self.status('parallel_started' if mode=='parallel' else 'serial_recovery_started',pid=child.pid,source_iteration=checkpoint['iteration'],output=str(output))
        return child,current

    def verify(self,child,meta,source_checkpoint):
        output=Path(meta['output']);deadline=time.time()+900
        while True:
            metric_file=output/'metrics.jsonl'
            if time.time()>deadline and (not metric_file.exists() or time.time()-metric_file.stat().st_mtime>600):
                raise RuntimeError('No parallel training progress within startup deadline')
            if child.poll() is not None:raise RuntimeError('Parallel trainer exited during startup')
            rows=(output/'metrics.jsonl').read_text().splitlines() if (output/'metrics.jsonl').exists() else []
            if len(rows)>=2 and (output/'last.pt').exists():
                c=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
                assert c['curriculum_mode']=='parallel' and c['parameter_max_difference']==0 and balance_passed(c)
                assert len(c['teams'])==1 and c['iteration']>source_checkpoint['iteration']
                import wandb
                try:
                    run=wandb.Api(timeout=30).run('mohismlab-robotics/q1-humanoid-kick-4v4/'+meta['wandb_run_id'])
                    summary=dict(run.summary)
                    if run.state=='running' and run.config.get('curriculum_mode')=='parallel' and summary.get('iteration',0)>meta['parent_iteration']:
                        meta.update(status='running',wandb_verified=True,latest_verified_iteration=c['iteration'],
                                    worker_pids=worker_pids(child.pid,self.project),parallel_transition_verified=True)
                        (self.project/'logs/active_training.json').write_text(json.dumps(meta,indent=2))
                        subprocess.run([self.python,'scripts/create_dashboard.py'],cwd=self.project,env=self.environment(meta),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90)
                        self.status('complete',pid=child.pid,iteration=c['iteration'],source_iteration=meta['parent_iteration'],
                                    allocations=c['curriculum']['allocations'],run_id=meta['wandb_run_id'],
                                    parameter_max_difference=0,checkpoint=str(output/'last.pt'))
                        return
                except Exception as error:
                    self.status('running_waiting_wandb_verification',pid=child.pid,error_type=type(error).__name__)
            time.sleep(15)

    def execute(self,meta):
        self.candidate_valid();self.stop(meta)
        source=Path(meta['output'])/'last.pt'
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        assert balance_passed(checkpoint),'Final checkpoint did not certify course 0'
        self.deploy()
        try:self.smoke(meta,source,checkpoint)
        except Exception:
            self.launch(meta,source,checkpoint,'serial');raise
        for worlds in dict.fromkeys((meta['num_envs'],3072,2048)):
            child,current=self.launch(meta,source,checkpoint,'parallel',worlds=worlds)
            try:
                self.verify(child,current,checkpoint)
                return
            except Exception:
                for pid in worker_pids(child.pid,self.project):os.kill(pid,signal.SIGTERM)
                deadline=time.time()+600
                while running(child.pid) and time.time()<deadline:time.sleep(5)
                if running(child.pid):raise RuntimeError('Parallel worker shutdown timed out')
                console=(Path(current['output'])/'console.log').read_text(errors='replace').lower()
                oom=any(text in console for text in ('out of memory','cuda_error_out_of_memory','failed to allocate'))
                if oom and worlds>2048:
                    self.status('retrying_smaller_parallel_batch',failed_worlds_per_gpu=worlds)
                    self.hardware(meta,set())
                    continue
                self.launch(meta,source,checkpoint,'serial')
                raise
        self.launch(meta,source,checkpoint,'serial')
        raise RuntimeError('All parallel memory configurations failed')



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--check-once',action='store_true')
    args=parser.parse_args();transition=Transition(args.root,args.candidate)
    lock=(transition.root/'runtime/parallel_transition.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        meta=transition.active();path=Path(meta['output'])/'last.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
        if meta.get('curriculum_mode')=='parallel':
            transition.status('already_parallel',pid=meta['pid']);return
        passed=balance_passed(checkpoint)
        transition.status('ready_for_transition' if passed else 'waiting_for_balance',
                          trainer_pid=meta['pid'],iteration=checkpoint.get('iteration'),
                          stage=checkpoint.get('curriculum',{}).get('stage'),
                          passed_windows=checkpoint.get('curriculum',{}).get('passed_windows'))
        if args.check_once:return
        if not running(meta['pid']):
            transition.status('trainer_stopped_before_transition');return
        if passed:
            try:
                transition.network_ok()
                transition.hardware(meta,set(worker_pids(meta['pid'],transition.project)))
            except Exception as error:
                transition.status('waiting_for_hardware_or_network',error_type=type(error).__name__)
                time.sleep(30);continue
            try:transition.execute(meta)
            except Exception as error:
                transition.status('transition_error',error_type=type(error).__name__,
                                  detail=str(error).replace(os.environ.get('WANDB_API_KEY','__unset__'),'[REDACTED]'))
            return
        time.sleep(30)


if __name__=='__main__':main()
