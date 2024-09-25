from typing import Tuple
import torch
import os
import numpy as np
import random

def seed(s: int):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.jit.script
def rotatepoint(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[...,3:4]
    q_xyz = q[...,:3]
    t = 2*torch.linalg.cross(q_xyz, v)
    return v + q_r * t + torch.linalg.cross(q_xyz, t)

@torch.jit.script
def heading_zup(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[...,:3])
    ref_dir[..., 0] = 1
    ref_dir = rotatepoint(q, ref_dir)
    return torch.atan2(ref_dir[...,1], ref_dir[...,0])

@torch.jit.script
def quatnormalize(q: torch.Tensor) -> torch.Tensor:
    q = (1-2*(q[...,3:4]<0).to(q.dtype))*q
    return q / q.norm(p=2, dim=-1, keepdim=True)

@torch.jit.script
def quatmultiply(q0: torch.Tensor, q1: torch.Tensor):
    x0, y0, z0, w0 = torch.unbind(q0, -1)
    x1, y1, z1, w1 = torch.unbind(q1, -1)
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return quatnormalize(torch.stack((x, y, z, w), -1))

@torch.jit.script
def quatconj(q: torch.Tensor):
    return torch.cat((-q[...,:3], q[...,-1:]), dim=-1)

@torch.jit.script
def axang2quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    # axis: n x 3
    # angle: n
    theta = (angle / 2).unsqueeze(-1)
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9))
    xyz = axis * torch.sin(theta)
    w = torch.cos(theta)
    return quatnormalize(torch.cat((xyz, w), -1))
 
@torch.jit.script
def wrap2pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat2axang(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = q[..., 3]

    sin = torch.sqrt(1 - w * w)
    mask = sin > 1e-5

    angle = 2 * torch.acos(w)
    angle = wrap2pi(angle)
    axis = q[..., 0:3] / sin.unsqueeze_(-1)

    z_axis = torch.zeros_like(axis)
    z_axis[..., -1] = 1

    angle = torch.where(mask, angle, z_axis[...,0])
    axis = torch.where(mask.unsqueeze_(-1), axis, z_axis)
    return axis, angle

@torch.jit.script
def quat2expmap(q: torch.Tensor) -> torch.Tensor:
    ax, ang = quat2axang(q)
    return ang.unsqueeze(-1)*ax

@torch.jit.script
def slerp(q0: torch.Tensor, q1: torch.Tensor, frac: torch.Tensor):
    c = q0[..., 3]*q1[..., 3] + q0[..., 0]*q1[..., 0] + \
        q0[..., 1]*q1[..., 1] + q0[..., 2]*q1[..., 2]
    q1 = torch.where(c.unsqueeze_(-1) < 0, -q1, q1)

    c = c.abs_()
    s = torch.sqrt(1.0 - c*c)
    t = torch.acos(c)

    c1 = torch.sin((1-frac)*t) / s
    c2 = torch.sin(frac*t) / s
    
    x = c1*q0[..., 0:1] + c2*q1[..., 0:1]
    y = c1*q0[..., 1:2] + c2*q1[..., 1:2]
    z = c1*q0[..., 2:3] + c2*q1[..., 2:3]
    w = c1*q0[..., 3:4] + c2*q1[..., 3:4]

    q = torch.cat((x, y, z, w), dim=-1)
    q = torch.where(s < 0.001, 0.5*q0+0.5*q1, q)
    q = torch.where(c >= 1, q0, q)
    return q

# @torch.jit.script
def dist2_p2seg_(p: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor):
    ab = s1 - s0
    ap = p - s0
    bp = p - s1

    near0 = (ap*ab).sum(-1) <= 0
    near1 = (bp*ab).sum(-1) >= 0
    near_line = ~torch.logical_or(near0, near1)
    
    dist2s0 = ap.square().sum(-1)
    dist2s1 = bp.square().sum(-1)
    dist2l = torch.linalg.cross(ab, ap).square().sum(-1) / ab.square().sum(-1)
    return dist2s0*near0 + dist2s1*near1 + dist2l*near_line

# @torch.jit.script
def closest_p2seg_(p: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor):
    ab = s1 - s0
    ap = p - s0
    bp = p - s1

    near0 = (ap*ab).sum(-1, keepdim=True) <= 0
    near1 = (bp*ab).sum(-1, keepdim=True) >= 0
    near_line = ~torch.logical_or(near0, near1)
    
    # dist2s0 = ap.square().sum(-1)
    # dist2s1 = bp.square().sum(-1)
    # dist2l = torch.linalg.cross(ab, ap).square().sum(-1) / ab.square().sum(-1)
    proj = s0 + (ap*ab).sum(-1, keepdim=True) * ab / ab.square().sum(-1, keepdim=True)

    return s0*near0 + s1*near1 + proj*near_line

# @torch.jit.script
def closest_seg2seg_(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor):
    r = b0 - a0
    v = b1 - b0
    u = a1 - a0

    ru = (r*u).sum(-1)
    rv = (r*v).sum(-1)
    uu = (u*u).sum(-1) #u.square().sum(-1)
    uv = (u*v).sum(-1)
    vv = (v*v).sum(-1) #v.square().sum(-1)

    det = uu*vv - uv.square()
    
    invalid = det < 1e-8
    valid = ~invalid
    det[invalid] = 1

    s = ru/uu * invalid + (((ru*uv-rv*uu)/det).clip(0, 1)*uv + ru)/uu * valid
    t = (s*uv - rv)/vv * invalid + (((ru*vv-rv*uv)/det).clip(0, 1)*uv - rv)/vv * valid

    p1x = a0+torch.clip(s, 0, 1).unsqueeze(-1)*u
    p2x = b0+torch.clip(t, 0, 1).unsqueeze(-1)*v
    return p1x, p2x

# @torch.jit.script
def dist2_seg2seg_(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor):
    # p1x, p2x = closest_seg2seg(a0, a1, b0, b1)

    r = b0 - a0
    v = b1 - b0
    u = a1 - a0

    ru = (r*u).sum(-1)
    rv = (r*v).sum(-1)
    uu = (u*u).sum(-1) #u.square().sum(-1)
    uv = (u*v).sum(-1)
    vv = (v*v).sum(-1) #v.square().sum(-1)

    det = uu*vv - uv.square()
    
    invalid = det < 1e-8
    valid = ~invalid
    det[invalid] = 1

    s = ru/uu * invalid + (((ru*uv-rv*uu)/det).clip(0, 1)*uv + ru)/uu * valid
    t = (s*uv - rv)/vv * invalid + (((ru*vv-rv*uv)/det).clip(0, 1)*uv - rv)/vv * valid

    p1x = a0+torch.clip(s, 0, 1).unsqueeze(-1)*u
    p2x = b0+torch.clip(t, 0, 1).unsqueeze(-1)*v
    return (p2x-p1x).square().sum(-1)

version_major, _ = torch.cuda.get_device_capability()
if version_major < 7:
    closest_p2seg = closest_p2seg_
    dist2_p2seg = dist2_p2seg_
    closest_seg2seg = closest_seg2seg_
    dist2_seg2seg = dist2_seg2seg_
else:
    # it sometime has errors when computing sum using jit.script with pytorch2.0
    closest_p2seg = torch.compile(closest_p2seg_)
    dist2_p2seg = torch.compile(dist2_p2seg_)
    closest_seg2seg = torch.compile(closest_seg2seg_)
    dist2_seg2seg = torch.compile(dist2_seg2seg_)
