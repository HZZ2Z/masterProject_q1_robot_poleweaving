"""Manage the server-side watchdog; stop the watchdog before intentionally stopping training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from q1_4v4.supervisor import TrainingSupervisor


class WatchdogCommand:
    def run(self):
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action',choices=('start','run','status','stop'))
        parser.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
        parser.add_argument('--root',default=os.environ.get('Q1_SERVER_ROOT'))
        parser.add_argument('--interval',type=int,default=15)
        parser.add_argument('--stall-seconds',type=int,default=600)
        options=parser.parse_args()
        if options.interval<1 or options.stall_seconds<60:parser.error('Invalid watchdog timing')
        project=Path(options.project).resolve();directory=project/'logs/watchdog';directory.mkdir(parents=True,exist_ok=True)
        pidfile=directory/'monitor.pid';stopfile=directory/'STOP'
        pid=int(pidfile.read_text()) if pidfile.exists() else None
        def alive():
            try:
                proc=Path('/proc',str(pid))
                cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
                return proc.stat().st_uid==os.getuid() and str(project).encode() in cmd and any(Path(arg.decode(errors='replace')).name=='watch_training.py' for arg in cmd if arg)
            except OSError:return False
        if options.action=='run':
            TrainingSupervisor(project,options.root,options.interval,options.stall_seconds).run();return
        if options.action=='stop':
            stopfile.touch()
            if alive():os.kill(pid,signal.SIGTERM)
            print(json.dumps(dict(watchdog_stopping=True,pid=pid,training_stopped=False)));return
        if options.action=='status':
            state=json.loads((directory/'state.json').read_text()) if (directory/'state.json').exists() else {}
            print(json.dumps(dict(state,supervisor_alive=alive()),ensure_ascii=False,indent=2));return
        if alive():print(json.dumps(dict(already_running=True,pid=pid)));return
        stopfile.unlink(missing_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'run','--project',str(project),
                 '--interval',str(options.interval),'--stall-seconds',str(options.stall_seconds)]
        if options.root:command.extend(('--root',options.root))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (directory/'console.log').open('ab') as stream:
            child=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(50):
            if pidfile.exists() and pidfile.read_text().strip()==str(child.pid):break
            if child.poll() is not None:raise RuntimeError('Watchdog failed to start; inspect logs/watchdog/console.log')
            time.sleep(.1)
        else:raise RuntimeError('Watchdog did not become ready; inspect logs/watchdog/console.log')
        print(json.dumps(dict(started=True,pid=child.pid,status_file=str(directory/'state.json'))))


if __name__=='__main__':WatchdogCommand().run()
