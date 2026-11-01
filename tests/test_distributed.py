"""Numerical checks for gradient averaging before clipping and shared moments."""
from types import SimpleNamespace
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from q1_4v4.distributed import Distributed, moments
from q1_4v4.gpu_learning import DeviceValueNorm,normalize_by_course


def worker(rank, store):
    os.environ['RANK']=str(rank)
    os.environ['WORLD_SIZE']='2'
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    context=Distributed('cpu')
    values=torch.arange(0,2,dtype=torch.float32) if rank==0 else torch.arange(2,5,dtype=torch.float32)
    mean,var=moments(values)
    torch.testing.assert_close(mean,torch.tensor(2.))
    torch.testing.assert_close(var,torch.tensor(2.))
    course_ids=torch.tensor([1,2] if rank==0 else [1,2,2])
    advantage=values.reshape(1,-1,1,1)
    normalized=normalize_by_course(advantage,torch.ones_like(advantage,dtype=torch.bool),course_ids)
    means=torch.tensor([1.,8/3]);std=torch.tensor([1.,(14/9)**.5])
    torch.testing.assert_close(normalized.flatten(),(values-means[course_ids-1])/(std[course_ids-1]+1e-5))
    normalizer=DeviceValueNorm('cpu');normalizer.update(values[:,None])
    torch.testing.assert_close(normalizer.running_mean,torch.tensor([.00002]))
    torch.testing.assert_close(normalizer.running_mean_sq,torch.tensor([.00006]))
    torch.manual_seed(4)
    actor=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
    reference=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
    reference.load_state_dict(actor.state_dict())
    policy=SimpleNamespace(actor=actor,critic=torch.nn.Linear(2,1))
    context.attach(policy)
    optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
    expected_optimizer=torch.optim.Adam(reference.parameters(),lr=.01)
    full=torch.arange(8,dtype=torch.float32).reshape(4,2)/4
    for _ in range(2):
        optimizer.zero_grad();expected_optimizer.zero_grad()
        actor(full[rank*2:rank*2+2]).square().mean().backward()
        reference(full).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),.1)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),.1)
        optimizer.step();expected_optimizer.step()
        for actual,expected in zip(actor.parameters(),reference.parameters()):
            torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
    assert context.check_parameters([(policy,None,None)])==0
    local=torch.tensor([[rank,0],[rank,1]],dtype=torch.float32) if rank else torch.empty((0,2))
    torch.testing.assert_close(context.rows(local),torch.tensor([[1.,0.],[1.,1.]]))
    assert context.objects(rank)==[0,1]
    context.check_health(lambda:None)
    def injected_failure():
        if rank==1:raise RuntimeError('injected EPA_HORIZON')
    try:
        context.check_health(injected_failure)
    except RuntimeError as error:
        assert 'rank1' in str(error) and 'EPA_HORIZON' in str(error)
    else:raise AssertionError('Every rank must stop on a single-rank physics failure')
    finally:context.close()
    assert not dist.is_initialized()
    context.close()  # Cleanup remains safe if an outer finally runs too.


def test_two_workers_match_global_batch_and_handle_empty_episodes(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'gloo_store'),),nprocs=2,join=True)


@pytest.mark.parametrize('fail',[False,True])
def test_runner_releases_process_group_even_on_initialization_failure(monkeypatch,fail):
    from q1_4v4 import gpu_runner
    closed=[]
    context=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(gpu_runner,'Distributed',lambda device:context)
    def body(*args):
        if fail:raise RuntimeError('initialization failed')
        return 'last.pt'
    monkeypatch.setattr(gpu_runner,'_run',body)
    options=SimpleNamespace(num_envs=1,horizon=1,epochs=1,iterations=1,save_interval=1,episode_steps=None,device='cuda:0')
    if fail:
        with pytest.raises(RuntimeError,match='initialization failed'):gpu_runner.run(options)
    else:assert gpu_runner.run(options)=='last.pt'
    assert closed==[True]
