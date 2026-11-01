"""Role-conditioned movement commands and gait shaping for continuous soccer."""
import torch


class MatchLocomotion:
    terms=('gait_velocity','gait_swing','gait_landing','gait_support')

    @staticmethod
    def commands(env,anchors,chase):
        pos=env.xpos[:,env.bodies]
        rot=env.xmat[:,env.bodies].reshape(env.count,8,3,3)
        delta=anchors-pos[:,:,:2]
        distance=torch.linalg.vector_norm(delta,dim=-1)
        moving=(distance>torch.where(chase,.45,.30))&env.active
        direction=delta/distance.clamp(min=.01)[:,:,None]
        world=torch.cat((direction,torch.zeros_like(distance[:,:,None])),-1)
        local=torch.einsum('nai,naij->naj',world,rot)
        yaw=torch.atan2(rot[:,:,1,0],rot[:,:,0,0])
        error=torch.atan2(torch.sin(torch.atan2(delta[:,:,1],delta[:,:,0])-yaw),
                          torch.cos(torch.atan2(delta[:,:,1],delta[:,:,0])-yaw))
        env.gait_command[:,:,:2]=.18*local[:,:,:2]*moving[:,:,None]
        env.gait_command[:,:,2]=(1.2*error).clamp(-.5,.5)*moving

    @staticmethod
    def rewards(features,dt):
        """Standing earns no tracking bonus when movement is commanded."""
        moving=features['moving'];healthy=features['healthy']
        mask=(moving&healthy).float()
        speed=features['speed'];yaw=features['yaw_command']
        linear=torch.exp(-(features['linear_error']/.12).square())-torch.exp(-(speed/.12).square())
        angular=torch.exp(-(features['angular_error']/.4).square())-torch.exp(-(yaw/.4).square())
        progress=features['forward_speed'].clamp(min=0)/speed.clamp(min=.03)
        activity=.25+.75*progress.clamp(max=1)
        valid=features['landing']&features['expected']&(features['landing'].sum(-1)==1)[:,:,None]
        valid&=(moving&healthy&(features['forward_speed']>.02))[:,:,None]
        valid&=features['slip']<.08
        return dict(
            gait_velocity=dt*(.35*linear+.10*angular)*mask,
            gait_swing=dt*.45*features['lift_credit']*activity*mask,
            gait_landing=.04*valid.float().sum(-1)+.08*(valid&features['alternating']).float().sum(-1),
            gait_support=-dt*.12*(features['flight'].float()+features['crossed'].float())*moving)
