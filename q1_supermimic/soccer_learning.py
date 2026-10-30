"""Experimental goal encoder training, optionally joint with the motor decoder.

The motor checkpoint is not certified as a kicking skill library. Downstream
the vision baseline keeps the decoder frozen. The explicitly configured joint
experiment trains the decoder and goal encoder together; this is a Q1 extension.
"""
import torch

from .soccer import motor_sensor
from .objectives import frozen, imagined_return, lambda_returns, clip_actor_gradients


def soccer_update(policy,world,critic,head,observe,camera,actor_optimizer,critic_optimizer,
                  batch,phase=0,horizon=32,gamma=.99,lam=.95,generator=None,train_decoder=False):
    state=batch['state'][0].detach()
    history=batch['history'][0].detach()
    visual=batch['visual'][0].detach()
    queue=batch['queue'][0].detach()
    delay,age=batch['delay'][0],batch['age'][0]
    context=batch['context'][0].detach()
    previous=history[:,-1,-22:]
    actor_optimizer.zero_grad(set_to_none=True)
    world.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
    rewards,terminations,inputs,values=[],[],[],[]
    fixed=[world,critic,policy.reference_encoder]
    if not train_decoder:fixed.append(policy.decoder)
    actor_parameters=list(policy.visual_encoder.parameters())
    if train_decoder:actor_parameters+=list(policy.decoder.parameters())
    decoder_before=policy.decoder[-1].weight.detach().clone()
    with frozen(*fixed):
        for t in range(horizon+1):
            phase_features=torch.nn.functional.one_hot(torch.full((len(state),),phase,device=state.device),4).to(state)
            critic_input=torch.cat((state,visual.flatten(1),phase_features),-1)
            inputs.append(critic_input.detach())
            values.append(critic(critic_input).squeeze(-1))
            if t==horizon:break
            action=policy.soccer(history.flatten(1),visual.flatten(1))
            following=world(state,action,context)
            reward,_,termination=head(state,following,action,previous,phase)
            rewards.append(reward);terminations.append(termination)
            features,queue=camera.advance(following,queue,delay,age+t+1,generator)
            visual=torch.cat((visual[:,1:],features.unsqueeze(1)),1)
            history=torch.cat((history[:,1:],motor_sensor(observe,following,action,generator).unsqueeze(1)),1)
            state,previous=following,action
        rewards=torch.stack(rewards);terminations=torch.stack(terminations);values=torch.stack(values)
        objective=imagined_return(rewards,terminations,values[-1],gamma).mean()
        if not torch.isfinite(objective):raise FloatingPointError('Non-finite soccer objective')
        (-objective).backward()
        visual_grad=torch.stack([p.grad.norm() for p in policy.visual_encoder.parameters() if p.grad is not None]).norm()
        decoder_grads=[p.grad.norm() for p in policy.decoder.parameters() if p.grad is not None]
        decoder_grad=torch.stack(decoder_grads).norm() if decoder_grads else state.new_zeros(())
        grad=clip_actor_gradients(actor_parameters)
    actor_optimizer.step()
    targets=lambda_returns(rewards.detach(),terminations.detach(),values.detach(),gamma,lam).flatten()
    inputs=torch.stack(inputs[:-1]).flatten(0,1)
    for _ in range(5):
        critic_optimizer.zero_grad(set_to_none=True)
        loss=(critic(inputs).squeeze(-1)-targets).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
        critic_optimizer.step()
    return dict(imagined_return=float(objective.detach()),actor_grad_norm=float(grad),
                visual_grad_norm=float(visual_grad),decoder_grad_norm=float(decoder_grad),
                decoder_output_weight_delta_norm=float((policy.decoder[-1].weight.detach()-decoder_before).norm()),
                critic_loss=float(loss.detach()))
