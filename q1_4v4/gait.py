"""Measured ground contacts and phase-conditioned gait shaping for Q1."""
import torch
import warp as wp
from .course_config import GAIT_COURSES,LOW_LEVEL

REWARD_REVISION = 6
METRIC_NAMES = ('slip_mps','phase_match','velocity_error','left_steps','right_steps',
                'clearance_m','weight_shifts','nonfoot_contact','alternations','phase_lift',
                'knee_lift','position_drift','heading_error','stance_width',
                'marching_progress','single_support','knee_signal_rate')


def marching_progress(kneelocal,neutral_knee,knee_angles,com_y,sin,expected):
    """Independent preparation signals; equal-leg crouching earns no progress."""
    def improvement(actual,target,scale):
        return torch.exp(-((actual-target)/scale).square())-torch.exp(-(target/scale).square())
    # Positive sine means left support/right swing, hence a negative left-right target.
    bend=improvement(knee_angles[...,0]-knee_angles[...,1],-.20*sin,.25)
    rise=improvement(kneelocal[...,0,2]-kneelocal[...,1,2],-.01*sin,.025)
    forward=improvement(kneelocal[...,0,0]-kneelocal[...,1,0],-.025*sin,.06)
    transfer=improvement(com_y,.03*sin,.03)
    # Recover the inherited rearward knee posture without rewarding common crouch.
    stance_offset=((kneelocal[...,0]-neutral_knee[...,0])*expected).sum(-1)/expected.sum(-1).clamp(min=1)
    posture=torch.tanh(stance_offset/.08).square()
    return 1.2*bend+.5*rise+.5*forward+.4*transfer-.2*posture


def marching_shaping(feetlocal, kneelocal, neutral_knee, clearance, targetheight,
                     contact, expected, drift, yawerror, linear_error, angular_error):
    """Pelvis-relative knee lift cannot be earned by raising the whole body."""
    rise = kneelocal[..., 2] - neutral_knee[..., 2]
    # A common rise of both knees from crouching is not a swing-knee lift.
    stance_rise=(rise*expected).sum(-1)/expected.sum(-1).clamp(min=1)
    rise=(rise-stance_rise.clamp(min=0)[...,None]).clamp(min=0)
    forward = (kneelocal[..., 0] - neutral_knee[..., 0]).clamp(min=0)
    target = .5*targetheight
    support = (contact & expected).float().sum(-1)/expected.sum(-1).clamp(min=1)
    knee = ((rise/target.clamp(min=.005)).clamp(0,1)
            *torch.exp(-((rise-target)/.02).square())
            *(forward/.025).clamp(0,1)*(~expected))
    # A lifted knee with a dragging foot does not count as a marching step.
    knee_credit = support*(knee*(clearance/.015).clamp(0,1)).sum(-1)
    width = feetlocal[...,0,1]-feetlocal[...,1,1]
    knee_width = kneelocal[...,0,1]-kneelocal[...,1,1]
    # Leave room for weight transfer; penalize sustained widening, not the sway itself.
    width_error = (width-.1754).abs().sub(.025).clamp(min=0)
    knee_error = (knee_width-.1817).abs().sub(.035).clamp(min=0)
    def cost(error,scale):
        x2=(error/scale).square()
        return x2/(1+x2)
    penalty = (.35*cost(drift,.10)+.15*cost(yawerror,.20)
               +.15*cost(linear_error,.10)+.10*cost(angular_error,.20)
               +.25*cost(width_error,.06)+.15*cost(knee_error,.06))
    return knee_credit, penalty, width


def marching_success(quality, landings, alternations, max_drift, max_heading, mean_knee):
    return (quality & (landings.amin(-1)>=4) & (alternations>=7)
            & (max_drift<.10) & (max_heading<.25) & (mean_knee>.15))


def phase_lift_credit(clearance, contact, expected, targetheight):
    """Dense lift progress paired with the correct supporting foot, symmetric in feet."""
    swing = ~expected
    progress = (clearance/targetheight.clamp(min=.005)).clamp(0,1)
    tracking = torch.exp(-((clearance-targetheight)/.02).square())
    support = (contact & expected).float().sum(-1)/expected.sum(-1).clamp(min=1)
    return support*(progress*tracking*swing).sum(-1)


def stepping_reward(dt, credit, contact, expected, linear_error, angular_error,
                    stance_slip, crossed, healthy, landing, alternating):
    """No standstill velocity bonus; reward real phase-conditioned leg motion."""
    double = expected.all(-1)
    stillness = torch.exp(-(linear_error/.1).square()-(angular_error/.2).square())
    swing_contact = (contact & ~expected).float().sum(-1)
    flight = ~contact.any(-1)
    reward = dt*((2.5+.5*stillness)*credit*healthy
        +.25*(double & contact.all(-1))-.75*swing_contact
        -.25*stance_slip-.3*crossed-1.*flight)
    # Only a single, correctly timed landing while upright earns an event reward.
    valid = landing & expected & (landing.sum(-1)==1)[...,None] & healthy[...,None]
    reward += .03*valid.sum(-1)+.17*(valid & alternating).sum(-1)
    return reward

@wp.kernel
def clear_ground(hit:wp.array3d[int],index:wp.array3d[int]):
    w,a,f=wp.tid()
    hit[w,a,f]=0
    index[w,a,f]=2147483647


@wp.kernel
def ground_contacts(n:wp.array[int],geoms:wp.array[wp.vec2i],worlds:wp.array[int],
                    dist:wp.array[float],owner:wp.array[int],body_owner:wp.array[int],floor:int,
                    hit:wp.array3d[int],index:wp.array3d[int],bad:wp.array2d[int]):
    c=wp.tid()
    if c<n[0] and dist[c]<=0.:
        a,b=geoms[c][0],geoms[c][1]
        g=int(-1)
        if a==floor:g=b
        elif b==floor:g=a
        if g>=0:
            k=owner[g];w=worlds[c]
            if k>=0:
                wp.atomic_max(hit,w,k//2,k%2,1)
                wp.atomic_min(index,w,k//2,k%2,c)
            elif body_owner[g]>=0:
                wp.atomic_max(bad,w,body_owner[g],1)


@wp.kernel
def ground_points(index:wp.array3d[int],points:wp.array[wp.vec3],out:wp.array4d[float]):
    w,a,f=wp.tid()
    c=index[w,a,f]
    if c<points.shape[0]:
        for j in range(3):out[w,a,f,j]=points[c][j]


def initialize(env):
    n,d=env.count,env.device
    env.gait_contact=torch.zeros((n,8,2),device=d,dtype=torch.int32)
    env.gait_contact_index=torch.full((n,8,2),2147483647,device=d,dtype=torch.int32)
    env.gait_bad_contact=torch.zeros((n,8),device=d,dtype=torch.int32)
    env.gait_point=torch.zeros((n,8,2,3),device=d)


def allocate(env):
    n,d=env.count,env.device
    z=lambda *s:torch.zeros(s,device=d)
    env.gait_command=z(n,8,3)
    env.gait_period=z(n,8)
    env.gait_offset=z(n,8)
    env.gait_height=z(n,8)
    env.gait_air=z(n,8,2)
    env.gait_peak=z(n,8,2)
    env.gait_landings=z(n,8,2)
    env.gait_last_landing=torch.full((n,8),-1,device=d,dtype=torch.long)
    env.gait_switches=z(n,8)
    env.gait_last_contact=torch.zeros((n,8,2),device=d,dtype=torch.bool)
    env.gait_match=z(n,8);env.gait_samples=z(n,8)
    env.gait_slip=z(n,8);env.gait_stance=z(n,8);env.gait_slip_bad=z(n,8)
    env.gait_command_distance=z(n,8);env.gait_actual_distance=z(n,8);env.gait_double=z(n,8)
    env.gait_error=z(n,8);env.gait_yaw_error=z(n,8)
    env.gait_flight=z(n,8);env.gait_flight_max=z(n,8)
    env.gait_nonfoot=z(n,8)
    env.gait_cross=z(n,8);env.gait_bad_clearance=z(n,8)
    env.gait_walk_mode=torch.zeros((n,8),device=d,dtype=torch.bool)
    env.gait_drift_max=z(n,8);env.gait_heading_max=z(n,8)
    env.gait_knee_credit=z(n,8)
    env.gait_previous_velocity=z(n,8,22)
    env.gait_stop_hold=z(n,8);env.gait_stop_count=z(n,8)
    env.gait_stop_latched=torch.zeros((n,8),device=d,dtype=torch.bool)
    env.gait_support_last=torch.full((n,8),-1,device=d,dtype=torch.long)
    env.gait_weight_count=z(n,8)
    # Four conservative sole corners from the official Q1 foot collision bounds.
    import yaml
    from .scene import ROOT
    cfg=yaml.safe_load((ROOT/'resources/Q1.yaml').read_text())
    bounds=cfg['asset']['foot_collision_bounds']
    l,h=bounds
    from .scene import PREFIXES
    env.knees=torch.tensor([[env.model.body(prefix+side+'_knee_link').id for side in ('left','right')] for prefix in PREFIXES],device=d)
    env.knee_dofs=torch.tensor([cfg['asset']['dof_names'].index(side+'_knee_joint') for side in ('left','right')],device=d)
    ref=env.reference.data
    root=env.model.body(PREFIXES[0]+'pelvis').id
    env.neutral_knee=env.tensor((ref.xpos[env.knees[0].cpu().numpy()]-ref.xpos[root])@ref.xmat[root].reshape(3,3))
    env.foot_roots=torch.as_tensor(env.model.body_rootid[env.feet.cpu().numpy()],device=d,dtype=torch.long)
    env.sole=env.tensor([[x,y,l[2]] for x in (l[0],h[0]) for y in (l[1],h[1])])


def reset(env,w,a):
    n=len(w)
    for name in ('gait_air','gait_peak','gait_landings','gait_switches','gait_match','gait_samples','gait_slip',
                 'gait_stance','gait_slip_bad','gait_error','gait_yaw_error','gait_flight','gait_flight_max',
                 'gait_cross','gait_bad_clearance','gait_previous_velocity','gait_stop_hold','gait_stop_count',
                 'gait_weight_count','gait_nonfoot','gait_command_distance','gait_actual_distance','gait_double',
                 'gait_drift_max','gait_heading_max','gait_knee_credit'):
        getattr(env,name)[w,a]=0
    env.gait_last_contact[w,a]=False;env.gait_stop_latched[w,a]=False
    env.gait_last_landing[w,a]=-1;env.gait_support_last[w,a]=-1
    env.gait_walk_mode[w,a]=False
    env.gait_period[w,a]=env.uniform((n,),.8,1.2)
    env.gait_offset[w,a]=env.uniform((n,),0.,1.)
    env.gait_height[w,a]=env.uniform((n,),.01,.03)
    if env.course=='step_in_place':env.gait_height[w,a]=env.uniform((n,),.02,.03)
    env.gait_contact[w,a]=0;env.gait_bad_contact[w,a]=0
    commands=env.gait_command
    commands[w,a]=0
    c=env.course
    if c in ('slow_walk','velocity_track','walk_turn','start_stop'):
        commands[w,a,0]=env.uniform((n,),.05,.15)
    if c=='velocity_track':
        # Command quadrants are resampled per episode; no privileged desired direction.
        choice=torch.randint(0,3,(n,),generator=env.generator,device=env.device)
        commands[w,a,0]=torch.where(choice==1,-commands[w,a,0],commands[w,a,0])
        commands[w,a,1]=torch.where(choice==2,env.uniform((n,),-.12,.12),0.)
        commands[w,a,0]*=(choice!=2)
    if c=='walk_turn':commands[w,a,2]=env.uniform((n,),-.35,.35)


def phase(env):
    return (env.agent_steps*env.dt/env.gait_period+env.gait_offset)%1.


def observations(env):
    p=phase(env)
    cmd=env.gait_command.clone()
    if env.course=='start_stop':cmd[:,:,:2]*=((env.agent_steps*env.dt%4.)<2.)[:,:,None]
    return torch.cat((cmd,torch.sin(2*torch.pi*p)[:,:,None],torch.cos(2*torch.pi*p)[:,:,None],
                      env.gait_height[:,:,None],env.gait_period[:,:,None]),-1)


def measure(env,before_foot):
    """Return [world,robot] shaping and end-of-episode quality criteria."""
    norm=lambda x:torch.linalg.vector_norm(x,dim=-1)
    n=env.count
    rot=env.xmat[:,env.bodies].reshape(n,8,3,3)
    pos=env.xpos[:,env.bodies]
    foot=env.xpos[:,env.feet]
    foot_rot=env.xmat[:,env.feet].reshape(n,8,2,3,3)
    corners=foot[:,:,:,None,:]+torch.einsum('nafij,kj->nafki',foot_rot,env.sole)
    clearance=corners[:,:,:,:,2].amin(-1).clamp(min=0)
    contact=env.gait_contact.bool()
    # MuJoCo cvel is in world axes about subtree CoM; shift to the actual sole patch.
    cvel=wp.to_torch(env.data.cvel)[:,env.feet]
    com=wp.to_torch(env.data.subtree_com)[:,env.foot_roots]
    patch_velocity=cvel[:,:,:,3:]+torch.cross(cvel[:,:,:,:3],env.gait_point-com,dim=-1)
    patch_slip=norm(patch_velocity[:,:,:,:2])
    # A flat foot turning about its centre still slides at the edges. Evaluate
    # the sole patch rather than only a single point that could be a pivot.
    corner_velocity=cvel[:,:,:,None,3:]+torch.cross(cvel[:,:,:,None,:3].expand_as(corners),corners-com[:,:,:,None],dim=-1)
    ground_corner=corners[:,:,:,:,2]<=.005
    corner_slip=norm(corner_velocity[:,:,:,:,:2])
    slip=torch.where(ground_corner.any(-1),(corner_slip*ground_corner).sum(-1)/ground_corner.sum(-1).clamp(min=1),patch_slip)
    env.foot_slip=slip
    v=env.qvel[:,env.rootv[:,None]+torch.arange(3,device=env.device)]
    vlocal=torch.einsum('nai,naij->naj',v,rot)
    yaw_velocity=env.qvel[:,env.rootv+5]
    p=phase(env);sin=torch.sin(2*torch.pi*p)
    expected=torch.stack((sin>=0,sin<0),-1)
    double=sin.abs()<.15
    expected=torch.where(double[:,:,None],True,expected)
    c=env.course
    cmd=env.gait_command.clone()
    targetdelta=env.targets[:,:,:2]-pos[:,:,:2]
    targetlocal=torch.einsum('nai,naij->naj',torch.cat((targetdelta,torch.zeros_like(targetdelta[:,:,:1])),-1),rot)
    yaw=torch.atan2(rot[:,:,1,0],rot[:,:,0,0])
    yawerror=torch.atan2(torch.sin(env.heading-yaw),torch.cos(env.heading-yaw))
    drift=norm(pos[:,:,:2]-env.start_positions[:,:,:2])
    env.gait_drift_max=torch.maximum(env.gait_drift_max,drift)
    env.gait_heading_max=torch.maximum(env.gait_heading_max,yawerror.abs())
    if c in ('move','backward','lateral','stop','approach'):
        if c=='approach':
            targetlocal=torch.einsum('nai,naij->naj',env.ball[:,None]-pos,rot)
        command_xy=.15*targetlocal[:,:,:2]/norm(targetlocal[:,:,:2]).clamp(min=.01)[:,:,None]
        distance=norm(targetlocal[:,:,:2])
        stop_radius=.40 if c=='approach' else .20
        command_xy*=((distance>stop_radius))[:,:,None]
        cmd[:,:,:2]=command_xy
    if c=='turn':cmd[:,:,2]=(yawerror*1.5).clamp(-.35,.35)*(yawerror.abs()>.15)
    if c=='start_stop':cmd[:,:,:2]*=((env.agent_steps*env.dt%4.)<2.)[:,:,None]
    env.gait_command.copy_(cmd) if c not in ('start_stop',) else None
    walking=(norm(cmd[:,:,:2])>.03)|(cmd[:,:,2].abs()>.05)|(c=='step_in_place')
    env.gait_walk_mode.copy_(walking)
    if c=='weight_shift':expected.fill_(True)
    elif c in GAIT_COURSES or c in ('teamplay','match'):expected=torch.where(walking[:,:,None],expected,True)
    else:expected.fill_(True)
    mismatch=(contact!=expected).float().mean(-1)
    # Event-based, debounced landings. An airborne foot must really clear the ground.
    env.gait_air=torch.where(contact,env.gait_air,env.gait_air+env.dt)
    env.gait_peak=torch.maximum(env.gait_peak,clearance)
    land=contact&~env.gait_last_contact&(env.gait_air>=.1)&(env.gait_peak>=.01)
    env.gait_landings+=land
    alternating=torch.zeros_like(land)
    for f in (0,1):
        event=land[:,:,f]&(land.sum(-1)==1)
        switch=event&(env.gait_last_landing>=0)&(env.gait_last_landing!=f)
        alternating[:,:,f]=switch
        env.gait_switches+=switch
        env.gait_last_landing=torch.where(event,f,env.gait_last_landing)
    env.gait_air=torch.where(contact,0.,env.gait_air)
    env.gait_peak=torch.where(contact,0.,env.gait_peak)
    env.gait_last_contact.copy_(contact)
    flight=~contact.any(-1)
    env.gait_flight=torch.where(flight,env.gait_flight+env.dt,0.)
    env.gait_flight_max=torch.maximum(env.gait_flight_max,env.gait_flight)
    env.gait_double+=contact.all(-1)
    env.gait_samples+=walking
    env.gait_match+=(1-mismatch)*walking
    env.gait_stance+=contact.sum(-1)
    env.gait_slip+=(slip*contact).sum(-1)
    env.gait_slip_bad+=((slip>.05)&contact).sum(-1)
    linear_error=norm(vlocal[:,:,:2]-cmd[:,:,:2])
    angular_error=(yaw_velocity-cmd[:,:,2]).abs()
    env.gait_error+=linear_error*walking
    env.gait_yaw_error+=angular_error*walking
    feetlocal=torch.einsum('nafi,naij->nafj',foot-pos[:,:,None],rot)
    # Signed left-right separation prevents crossing; upper bounds avoid extreme splits.
    separation=feetlocal[:,:,0,1]-feetlocal[:,:,1,1]
    kneelocal=torch.einsum('nafi,naij->nafj',env.xpos[:,env.knees]-pos[:,:,None],rot)
    knee_separation=kneelocal[:,:,0,1]-kneelocal[:,:,1,1]
    crossed=(separation<.045)|(separation>.30)|(knee_separation<.03)|(knee_separation>.26)
    if c=='step_in_place':
        # Q1 feet are 6.59 cm wide and shank collision cylinders 7.6 cm across.
        crossed=(separation<.09)|(separation>.23)|(knee_separation<.09)|(knee_separation>.24)
    env.gait_cross+=crossed
    env.gait_nonfoot+=env.gait_bad_contact.bool()
    env.gait_bad_clearance+=((clearance>.08).any(-1)&walking)
    ratio=env.gait_match/env.gait_samples.clamp(min=1)
    slipmean=env.gait_slip/env.gait_stance.clamp(min=1)
    badratio=env.gait_slip_bad/env.gait_stance.clamp(min=1)
    quality=(ratio>=.8)&(slipmean<=.05)&(badratio<=.1)&(env.gait_flight_max<=.1)&(
        env.gait_cross/env.agent_steps.clamp(min=1)<=.05)&(env.gait_bad_clearance/env.agent_steps.clamp(min=1)<=.05)&(env.gait_nonfoot==0)
    command_speed=norm(cmd[:,:,:2])
    env.gait_command_distance+=command_speed*env.dt
    env.gait_actual_distance+=(vlocal[:,:,:2]*cmd[:,:,:2]).sum(-1)/command_speed.clamp(min=.01)*env.dt
    velocity_ok=(env.gait_error/env.gait_samples.clamp(min=1)<= (.05 if c=='slow_walk' else .08)) & (
        env.gait_actual_distance>=.5*env.gait_command_distance)
    quality &= env.gait_samples>=100
    targetheight=env.gait_height[:,:,None]*torch.sin(2*torch.pi*p[:,:,None]).abs()
    swing=~expected
    lift=torch.exp(-((clearance-targetheight)/.015).square())*swing
    stance_slip=((slip/.05).square().clamp(max=9)*contact).sum(-1)
    credit=phase_lift_credit(clearance,contact,expected,targetheight)
    progress=torch.zeros_like(credit)
    single_support=torch.zeros((),device=env.device)
    knee_signal_rate=torch.zeros((),device=env.device)
    reward=env.dt*(1.2*torch.exp(-(linear_error/.10).square())+.7*torch.exp(-(angular_error/.20).square())
        +.7*(1-mismatch)+.4*lift.sum(-1)-.25*stance_slip-.3*crossed-.3*flight)
    if c=='step_in_place':
        healthy=(rot[:,:,2,2]>.8)&(pos[:,:,2]>.30)&(pos[:,:,2]<.52)&~env.gait_bad_contact.bool()
        knee_credit,stationary_penalty,_=marching_shaping(feetlocal,kneelocal,env.neutral_knee,
            clearance,targetheight,contact,expected,drift,yawerror,linear_error,angular_error)
        knee_credit*=healthy
        env.gait_knee_credit+=knee_credit
        whole_com=wp.to_torch(env.data.subtree_com)[:,env.bodies]
        comlocal=torch.einsum('nai,naij->naj',whole_com-foot.mean(-2),rot)
        knee_angles=env.qpos[:,env.qidx][:,:,env.knee_dofs]
        progress=marching_progress(kneelocal,env.neutral_knee,knee_angles,comlocal[:,:,1],sin,expected)
        progress*=.5*(healthy&contact.any(-1))
        reward=stepping_reward(env.dt,credit,contact,expected,linear_error,angular_error,
                              stance_slip,crossed,healthy,land,alternating)
        reward+=env.dt*(.8*knee_credit+progress-stationary_penalty)
        single=~expected.all(-1)
        single_support=((contact==expected).all(-1)&single).float().sum()/single.sum().clamp(min=1)
        knee_signal_rate=(knee_credit>1e-8).float().mean()
    qv=env.qvel[:,env.vidx]
    reward-=env.dt*.002*((qv-env.gait_previous_velocity)/env.dt).square().mean(-1).clamp(max=25)
    env.gait_previous_velocity.copy_(qv)
    # Small pose prior on the upper body; legs remain free to execute the gait.
    reward-=env.dt*.1*(env.qpos[:,env.qidx][:,:,12:]-env.neutral[12:]).square().mean(-1)
    objective=quality&velocity_ok
    if c=='step_in_place':
        objective=marching_success(quality,env.gait_landings,env.gait_switches,
            env.gait_drift_max,env.gait_heading_max,env.gait_knee_credit/env.agent_steps.clamp(min=1))
    if c=='slow_walk':objective&=env.gait_landings.sum(-1)>=6
    if c=='walk_turn':objective&=env.gait_yaw_error/env.gait_samples.clamp(min=1)<=.15
    if c=='start_stop':
        stopped=(~walking)&(norm(vlocal[:,:,:2])<.1)
        env.gait_stop_hold=torch.where(stopped,env.gait_stop_hold+env.dt,0.)
        achieved=(env.gait_stop_hold>=1)&~env.gait_stop_latched
        env.gait_stop_count+=achieved
        env.gait_stop_latched=torch.where(walking,False,env.gait_stop_latched|achieved)
        objective&=env.gait_stop_count>=3
    if c=='weight_shift':
        # Move the measured subtree CoM toward alternating inner support locations.
        whole_com=wp.to_torch(env.data.subtree_com)[:,env.bodies]
        center=foot.mean(-2)
        comlocal=torch.einsum('nai,naij->naj',whole_com-center,rot)
        lateral_target=.025*torch.sin(2*torch.pi*p)
        error=(comlocal[:,:,1]-lateral_target).abs()
        near_peak=sin.abs()>.8
        desired=torch.where(sin>0,0,1)
        valid=near_peak&(error<.015)&contact.all(-1)&(slip.amax(-1)<.05)
        switch=valid&(env.gait_support_last>=0)&(env.gait_support_last!=desired)
        env.gait_weight_count+=switch
        env.gait_support_last=torch.where(valid,desired,env.gait_support_last)
        reward=env.dt*(1.5*torch.exp(-(error/.015).square())-.25*stance_slip+.3*contact.all(-1))
        reward-=env.dt*.2*(yaw_velocity/.2).square().clamp(max=4)
        objective=(env.gait_weight_count>=4)&(slipmean<=.05)&(badratio<=.1)&(env.gait_nonfoot==0)&(
            env.gait_double/env.agent_steps.clamp(min=1)>=.95)&(norm(pos[:,:,:2]-env.start_positions[:,:,:2])<.15)&(yawerror.abs()<.2)
    if c in ('teamplay','match'):
        healthy=(rot[:,:,2,2]>.85)&(pos[:,:,2]>.36)&(pos[:,:,2]<.50)&~env.gait_bad_contact.bool()
        forward=(vlocal[:,:,:2]*cmd[:,:,:2]).sum(-1)/command_speed.clamp(min=.03)
        env.match_gait_features=dict(moving=walking,healthy=healthy,speed=command_speed,
            yaw_command=cmd[:,:,2],linear_error=linear_error,angular_error=angular_error,
            forward_speed=forward,lift_credit=credit,landing=land,expected=expected,
            alternating=alternating,slip=slip,flight=flight,crossed=crossed)
    env.gait_metrics={'slip_mps':slipmean.mean(),'phase_match':ratio.mean(),
        'velocity_error':(env.gait_error/env.gait_samples.clamp(min=1)).mean(),
        'left_steps':env.gait_landings[:,:,0].mean(),'right_steps':env.gait_landings[:,:,1].mean(),
        'nonfoot_contact':(env.gait_nonfoot>0).float().mean(),'clearance_m':clearance.mean(),'weight_shifts':env.gait_weight_count.mean(),
        'alternations':env.gait_switches.mean(),'phase_lift':credit.mean(),
        'knee_lift':(env.gait_knee_credit/env.agent_steps.clamp(min=1)).mean(),
        'position_drift':drift.mean(),'heading_error':yawerror.abs().mean(),'stance_width':separation.mean(),
        'marching_progress':progress.mean(),'single_support':single_support,'knee_signal_rate':knee_signal_rate}
    return reward,objective,quality
