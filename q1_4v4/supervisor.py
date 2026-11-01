"""Server-resident supervision of one explicitly authorized training run."""
import base64
import csv
import fcntl
import json
import netrc
import os
from pathlib import Path
import pickle
import signal
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse
import shutil


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(path)


class CheckpointStore:
    invalid=(OSError,RuntimeError,ValueError,EOFError,KeyError,TypeError,AttributeError,pickle.UnpicklingError)

    def __init__(self,output,directory):
        self.output=Path(output);self.directory=Path(directory)

    @staticmethod
    def read(path):
        import torch
        torch.set_num_threads(1)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(checkpoint,dict) or checkpoint.get('backend')!='warp' or not isinstance(checkpoint.get('iteration'),int) or checkpoint['iteration']<0:
            raise ValueError('Not a resumable Warp checkpoint')
        if checkpoint.get('parameter_max_difference',0.)!=0:
            raise ValueError('Checkpoint contains divergent distributed policies')
        if not checkpoint.get('teams'):raise ValueError('Checkpoint has no policy')
        for team in checkpoint['teams']:
            for name in ('actor','critic'):
                if not team.get(name):raise ValueError('Checkpoint has no '+name)
                if any(not torch.isfinite(value).all().item() for value in team[name].values()):
                    raise ValueError('Nonfinite checkpoint parameters')
            if any(not torch.isfinite(value).all().item() for value in team.get('value_normalizer',{}).values()):
                raise ValueError('Nonfinite ValueNorm state')
            for name in ('actor_optimizer','critic_optimizer'):
                if 'state' not in team.get(name,{}) or 'param_groups' not in team[name]:
                    raise ValueError('Checkpoint has no optimizer state')
                for state in team[name]['state'].values():
                    for value in state.values():
                        if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                            raise ValueError('Nonfinite optimizer state')
        amp=checkpoint.get('amp')
        if amp:
            if checkpoint.get('amp_parameter_max_difference',0.)!=0:
                raise ValueError('Checkpoint contains divergent AMP parameters')
            tensors=list(amp['model'].values())
            tensors.extend(value for state in amp['optimizer']['state'].values() for value in state.values() if torch.is_tensor(value))
            if any(not torch.isfinite(value).all().item() for value in tensors):
                raise ValueError('Nonfinite AMP checkpoint state')
        return checkpoint

    def refresh(self):
        temporary=self.directory/'last_good.pt.tmp'
        shutil.copyfile(self.output/'last.pt',temporary)
        checkpoint=self.read(temporary)
        temporary.replace(self.directory/'last_good.pt')
        atomic_json(self.directory/'last_good.json',dict(output=str(self.output),iteration=checkpoint['iteration']))
        return checkpoint['iteration']

    def resume(self):
        errors=[]
        for source in (self.output/'last.pt',self.directory/'last_good.pt'):
            try:
                if source.name=='last_good.pt':
                    manifest=json.loads((self.directory/'last_good.json').read_text())
                    if manifest['output']!=str(self.output):raise ValueError('Backup belongs to another training output')
                temporary=self.directory/'resume.pt.tmp';shutil.copyfile(source,temporary)
                checkpoint=self.read(temporary);temporary.replace(self.directory/'resume.pt')
                return self.directory/'resume.pt',checkpoint
            except self.invalid as error:
                errors.append(f'{source.name}: {type(error).__name__}')
        raise RuntimeError('No valid checkpoint: '+'; '.join(errors))


class ProcessRegistry:
    def __init__(self,output):self.output=str(output)

    def inspect(self,pid):
        try:
            proc=Path('/proc')/str(pid)
            if proc.stat().st_uid!=os.getuid():return None
            stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
            args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            text=[arg.decode(errors='replace') for arg in args if arg]
            if stat[0]=='Z' or self.output not in text:return None
            if not any(Path(arg).name=='train_distributed.py' for arg in text):return None
            return dict(pid=int(pid),start_ticks=stat[19])
        except (OSError,IndexError):return None

    def workers(self,meta):
        pids=set(meta.get('worker_pids',[]))
        try:
            path=Path('/proc',str(meta['pid']),'task',str(meta['pid']),'children')
            pids.update(int(pid) for pid in path.read_text().split())
        except OSError:pass
        if self.inspect(meta['pid']) is None:
            pids.update(int(path.name) for path in Path('/proc').iterdir() if path.name.isdecimal())
        return [item for pid in sorted(pids) if (item:=self.inspect(pid)) is not None]

    def signal(self,identity,number):
        if identity and self.inspect(identity['pid'])==identity:
            try:os.kill(identity['pid'],number)
            except ProcessLookupError:pass


class ResourceGuard:
    def __init__(self,root):self.root=Path(root)

    def environment(self,meta):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(meta['gpu_uuids']),
                   WANDB_RUN_ID=meta['wandb_run_id'],WANDB_RESUME='must',
                   OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NCCL_SHM_DISABLE='1',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        host=urlparse(env['WANDB_BASE_URL']).hostname
        key=env.get('WANDB_API_KEY')
        if not key:
            credentials=netrc.netrc(env['NETRC']).authenticators(host)
            key=credentials[2] if credentials else None
        if not key:raise RuntimeError('W&B credentials unavailable')
        env['WANDB_API_KEY']=key
        proxy=json.loads((self.root/'runtime/wandb_proxy.json').read_text())['url']
        for name in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[name]=proxy
        env['NO_PROXY']=env['no_proxy']='localhost,127.0.0.1'
        return env

    @staticmethod
    def query(arguments):
        return subprocess.check_output(['nvidia-smi',*arguments],text=True,timeout=10)

    def available(self,meta):
        allowed=set(meta['gpu_uuids'])
        devices=list(csv.reader(self.query(['--query-gpu=uuid,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits']).splitlines()))
        found=set()
        for uuid,memory,utilization in devices:
            uuid=uuid.strip()
            if uuid not in allowed:continue
            found.add(uuid)
            if int(memory.strip())>512 or int(utilization.strip())>5:
                return False,'GPU not idle: '+uuid
        if found!=allowed:return False,'An assigned GPU is unavailable'
        applications=self.query(['--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])
        for line in csv.reader(applications.splitlines()):
            if len(line)>=2 and line[1].strip() in allowed:return False,'Assigned GPU has a compute process'
        return True,'ready'

    @staticmethod
    def network(env):
        endpoint=env['WANDB_BASE_URL'].rstrip('/')+'/graphql'
        query={'query':'query { project(name:"q1-humanoid-kick-4v4",entityName:"mohismlab-robotics") { name } }'}
        request=urllib.request.Request(endpoint,data=json.dumps(query).encode(),headers={
            'Content-Type':'application/json','Authorization':'Basic '+base64.b64encode(('api:'+env['WANDB_API_KEY']).encode()).decode()})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':env['HTTP_PROXY'],'https':env['HTTPS_PROXY']}))
        with opener.open(request,timeout=10) as response:result=json.load(response)
        if result.get('errors') or not result.get('data',{}).get('project'):
            raise RuntimeError('W&B project connectivity check failed')


class TrainingSupervisor:
    manual_states={'stopped','paused','completed','finished'}

    def __init__(self,project,root=None,interval=15,stall_seconds=600,grace_seconds=90):
        self.project=Path(project).resolve();self.root=Path(root or self.project.parent).resolve()
        self._confined(self.project)
        self.directory=self.project/'logs/watchdog';self.directory.mkdir(parents=True,exist_ok=True)
        self.directory.chmod(0o700)
        self.metadata=self.project/'logs/active_training.json'
        self.interval=interval;self.stall_seconds=stall_seconds;self.grace_seconds=grace_seconds
        self.guard=ResourceGuard(self.root);self.stop_event=threading.Event()
        self.state=dict(enabled=True,status='starting',pid=os.getpid(),restarts=0,interval_seconds=interval)
        self.pinned=None;self.generation=None;self.known={};self.shutdown_at=None
        self.started_at=time.time();self.next_restart=0.;self.backoff=15.;self.last_backup=0.
        self.child=None;self.store=None;self.registry=None

    def _confined(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Path outside authorized workspace')
        return path

    def event(self,name,**values):
        record=dict(time=time.time(),event=name,**values)
        with (self.directory/'events.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)

    def publish(self,**values):
        self.state.update(values,heartbeat=time.time())
        atomic_json(self.directory/'state.json',self.state)

    def metadata_read(self):
        meta=json.loads(self.metadata.read_text())
        output=self._confined(meta['output']);self._confined(meta['console_log'])
        if not meta.get('wandb_run_id') or not meta.get('gpu_uuids'):raise ValueError('Incomplete training identity')
        identity=(meta['wandb_run_id'],str(output))
        if self.pinned and identity!=self.pinned:
            self.publish(status='superseded',enabled=False);self.stop_event.set();return None
        self.pinned=identity
        if self.store is None:
            self.store=CheckpointStore(output,self.directory);self.registry=ProcessRegistry(output)
            self.state.update(run_id=meta['wandb_run_id'],output=str(output))
        return meta

    def _adopt(self,meta,manager,workers):
        self.generation=manager;self.known={item['pid']:item for item in workers}
        self.started_at=time.time();self.shutdown_at=None;self.last_backup=0.
        self.event('adopted_training',manager=manager['pid'],workers=list(self.known))

    @staticmethod
    def replace_argument(command,name,value):
        result=list(command)
        if name in result:result[result.index(name)+1]=str(value)
        else:result.extend((name,str(value)))
        return result

    def _restart(self,meta,now):
        ready,reason=self.guard.available(meta)
        if not ready:
            self.publish(status='waiting_gpu',last_error=reason);return
        env=self.guard.environment(meta)
        self.guard.network(env)
        snapshot,checkpoint=self.store.resume();iteration=checkpoint['iteration']
        target=int(meta.get('target_iteration',20000))
        if iteration>=target:
            self.publish(status='completed',enabled=False,last_iteration=iteration);self.stop_event.set();return
        if (self.directory/'STOP').exists() or self.stop_event.is_set():return
        command=list(meta['command'])
        if '--start-course' in command:
            index=command.index('--start-course');del command[index:index+2]
        command=self.replace_argument(command,'--checkpoint',snapshot)
        command=self.replace_argument(command,'--iterations',target-iteration)
        archive=self.store.output/'failures'/('watchdog_'+str(time.time_ns()))
        failures=list(self.store.output.glob('physics_failure*.json'))
        if failures:
            archive.mkdir(parents=True)
            for failure in failures:failure.rename(archive/failure.name)
        with Path(meta['console_log']).open('ab') as stream:
            stream.write(f'\nWatchdog resuming checkpoint iteration {iteration}\n'.encode());stream.flush()
            self.child=subprocess.Popen(command,cwd=self.project,env=env,stdin=subprocess.DEVNULL,
                                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        meta.update(pid=self.child.pid,worker_pids=[],command=command,checkpoint=str(snapshot),
                    parent_iteration=iteration,status='starting',wandb_resume='must',
                    restarted_at=time.time(),watchdog_pid=os.getpid(),watchdog_enabled=True)
        for key in ('exit_reason','failed_iteration','stopped_at','last_completed_iteration','alive_worker_pids'):
            meta.pop(key,None)
        atomic_json(self.metadata,meta)
        self.generation=self.registry.inspect(self.child.pid);self.known={};self.started_at=now
        self.shutdown_at=None;self.state['restarts']+=1
        self.publish(status='starting_training',last_iteration=iteration,last_restart=now,
                     manager_pid=self.child.pid,last_error=None)
        self.event('restarted_training',checkpoint_iteration=iteration,manager=self.child.pid)

    def tick(self):
        now=time.time();meta=self.metadata_read()
        if meta is None:return
        if (self.directory/'STOP').exists():self.stop_event.set();return
        status=meta.get('status','')
        if status in self.manual_states or status.startswith('stopping'):
            self.publish(status='manual_stop',manager_pid=meta.get('pid'));return
        manager=self.registry.inspect(meta['pid']);workers=self.registry.workers(meta)
        if manager and manager!=self.generation:self._adopt(meta,manager,workers)
        for item in workers:self.known.setdefault(item['pid'],item)
        workers=[item for item in workers if self.known.get(item['pid'])==item]
        checkpoint_path=self.store.output/'last.pt'
        age=now-checkpoint_path.stat().st_mtime if checkpoint_path.exists() else now-self.started_at
        expected=int(meta.get('world_size',len(meta['gpu_uuids'])))
        startup=status=='starting' and now-self.started_at<300
        if not self.shutdown_at and manager and (len(workers)==expected or startup) and (startup or age<self.stall_seconds):
            if checkpoint_path.exists() and now-self.last_backup>=60:
                try:
                    iteration=self.store.refresh();self.state['verified_iteration']=iteration
                    self.state['last_iteration']=iteration;self.last_backup=now
                    if iteration>=int(meta.get('target_iteration',20000)):
                        self.publish(status='completed',enabled=False);self.stop_event.set();return
                    if iteration>meta.get('parent_iteration',0):
                        meta.update(status='running',worker_pids=[item['pid'] for item in workers],
                                    watchdog_enabled=True,watchdog_pid=os.getpid(),latest_verified_iteration=iteration)
                        atomic_json(self.metadata,meta)
                    if now-self.started_at>300:self.backoff=15.
                except CheckpointStore.invalid as error:
                    self.event('checkpoint_backup_failed',error=type(error).__name__)
            self.publish(status='watching',manager_pid=meta['pid'],alive_workers=len(workers),
                         expected_workers=expected,checkpoint_age_seconds=round(age,1))
            return
        if self.shutdown_at is None:
            self.shutdown_at=now
            self.event('training_failure_detected',manager_alive=bool(manager),alive_workers=len(workers),checkpoint_age_seconds=round(age,1))
        for item in self.known.values():
            self.registry.signal(item,signal.SIGKILL if now-self.shutdown_at>self.grace_seconds else signal.SIGTERM)
        if workers:
            self.publish(status='stopping_failed_training',alive_workers=len(workers));return
        if manager:
            self.registry.signal(self.generation,signal.SIGTERM)
            if now-self.shutdown_at>self.grace_seconds:self.registry.signal(self.generation,signal.SIGKILL)
            self.publish(status='stopping_failed_manager');return
        if self.child:self.child.poll()
        if now<self.next_restart:
            self.publish(status='retry_wait',next_retry_at=self.next_restart);return
        try:
            self._restart(meta,now)
            self.next_restart=now+self.backoff;self.backoff=min(self.backoff*2,300.)
        except Exception as error:
            self.next_restart=now+self.backoff;self.backoff=min(self.backoff*2,300.)
            self.publish(status='restart_wait',last_error=f'{type(error).__name__}: {error}',next_retry_at=self.next_restart)
            self.event('restart_deferred',error=f'{type(error).__name__}: {error}')

    def run(self):
        lock=(self.directory/'monitor.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This project already has a training supervisor')
        os.umask(0o077)
        (self.directory/'monitor.pid').write_text(str(os.getpid()))
        for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda *_:self.stop_event.set())
        self.event('monitor_started',interval_seconds=self.interval)
        try:
            while not self.stop_event.is_set():
                try:self.tick()
                except Exception as error:
                    self.publish(status='check_error',last_error=f'{type(error).__name__}: {error}')
                    self.event('check_error',error=f'{type(error).__name__}: {error}')
                self.stop_event.wait(self.interval)
        finally:
            self.publish(enabled=False,status=self.state['status'] if self.state['status'] in ('completed','superseded') else 'stopped')
            (self.directory/'monitor.pid').unlink(missing_ok=True)
            self.event('monitor_stopped');lock.close()
