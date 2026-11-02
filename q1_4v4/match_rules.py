"""Penalty-area restrictions and physics-time last-touch attribution."""
import torch


class MatchRules:
    def __init__(self, count, device):
        self.last_touch_robot=torch.full((count,),-1,device=device,dtype=torch.long)
        self.last_touch_team=torch.full_like(self.last_touch_robot,-1)
        self.last_touch_mask=torch.zeros((count,8),device=device,dtype=torch.bool)

    def reset_worlds(self, ids):
        self.last_touch_robot[ids]=-1;self.last_touch_team[ids]=-1;self.last_touch_mask[ids]=False

    @staticmethod
    def penalty_area(positions):
        x,y=positions[...,0].abs(),positions[...,1].abs()
        field_player=(torch.arange(8,device=positions.device)%4)!=0
        return (x>=4.8)&(x<=7.)&(y<=2.8)&field_player

    def update_touches(self, times):
        latest=times.amax(-1,keepdim=True)
        self.last_touch_mask.copy_((times==latest)&(latest>=0))
        count=self.last_touch_mask.sum(-1)
        robot=self.last_touch_mask.long().argmax(-1)
        self.last_touch_robot.copy_(torch.where(count==1,robot,-1))
        teams=self.last_touch_mask.reshape(-1,2,4).any(-1)
        self.last_touch_team.copy_(torch.where(teams.sum(-1)==1,teams.long().argmax(-1),-1))

    def out_penalty(self, out):
        return -out[:,None].float()*self.last_touch_mask/self.last_touch_mask.sum(-1,keepdim=True).clamp(min=1)
