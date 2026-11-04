import torch
from multiview.roi_context import ExperimentGHead

def test_shapes_identity_gradient():
    torch.manual_seed(0)
    b,c=2,32
    head=ExperimentGHead(c,dim=16,geometry_weight=0)
    center=torch.randn(b,c,15,15)
    side=torch.randn(b,c,15,15)
    cb=torch.tensor([[[10.,10.,20.,20.],[30.,30.,40.,40.]],[[10.,10.,20.,20.],[30.,30.,40.,40.]]])
    sb=cb.clone()
    cm=torch.tensor([[True,True],[True,False]])
    sm=torch.tensor([[True,True],[True,False]])
    theta0=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]).repeat(b,1,1)
    centers=(sb[...,:2]+sb[...,2:])/2
    out=head(theta0,center,side,cb,sb,cm,sm,(60,60),centers)
    assert out['theta'].shape==(b,2,3)
    assert torch.equal(out['theta'],theta0)
    assert out['attention'].shape==(b,2,3)
    loss=out['theta'].sum()
    loss.backward()
    assert head.refine.net[-1].weight.grad is not None
    assert head.refine.net[-1].weight.grad.abs().sum()>0

if __name__=='__main__':
    test_shapes_identity_gradient()
    print('PASS: shape, identity, gradient')
