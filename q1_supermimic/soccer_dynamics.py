"""Ball-specific physical supervision for the joint soccer experiment."""
from .dynamics import Dynamics


class SoccerDynamics(Dynamics):
    def __init__(self,model,config,**kwargs):
        super().__init__(model,**kwargs)
        self.ball_q=int(model.joint('ball_free').qposadr[0])
        self.ball_v=int(model.joint('ball_free').dofadr[0])+model.nq
        self.ball_position_weight=config.get('ball_position_loss_weight',0.)
        self.ball_velocity_weight=config.get('ball_velocity_loss_weight',0.)

    def error(self,predicted,actual):
        result=super().error(predicted,actual)
        for address,weight in ((self.ball_q,self.ball_position_weight),(self.ball_v,self.ball_velocity_weight)):
            error=(predicted[...,address:address+3]-actual[...,address:address+3])/self.scale[address:address+3]
            result=result+weight*error.square().mean()
        return result
