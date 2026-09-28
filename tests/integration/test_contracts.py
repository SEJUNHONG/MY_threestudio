"""CPU regression tests of real method bodies, with external CUDA/models replaced.
AST loading isolates methods from threestudio's eager native imports; this is NOT
an end-to-end runtime or GPU test.
"""
import ast
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)

def load_file(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

u = load_file('threestudio/utils/integration.py')

class Base(torch.nn.Module):
    @dataclasses.dataclass
    class Config:
        pass

def load_class(path, name, **extra):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in nodes:
        node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(tree)
    env = dict(torch=torch, F=F, np=np, dataclass=dataclasses.dataclass, field=dataclasses.field, BaseLift3DSystem=Base, BaseModule=Base, Rasterizer=Base)
    env.update({k:v for k,v in vars(u).items() if not k.startswith("__")})
    env["__name__"] = __name__
    env.update(extra)
    exec(compile(tree, path, 'exec'), env)
    return env[name], env

def test_view_order_and_invalid_batch():
    values = torch.tensor([1,1,1,1,2,2,2,2])
    result = u.append_group_view(values, 4)
    assert result.tolist() == [1]*5+[2]*5
    assert torch.equal(u.remove_group_view(result,4), values)
    with pytest.raises(ValueError): u.validate_view_batch(5,4)

def test_background_preserves_color_and_all_views():
    bg = torch.zeros(4,2,3,3);bg[...,0]=1;bg[2:,...,1]=1
    assert np.allclose(u.background_rgb(bg), [255,127.5,0])
    with pytest.raises(ValueError):u.background_rgb(None)

def test_rasterizer_abi_error():
    with pytest.raises(RuntimeError, match='four tensors'):u.unpack_rasterizer((None,None))
    with pytest.raises(ValueError):u.unpack_rasterizer((torch.zeros(3,2,2),torch.ones(5),torch.zeros(2,2),torch.zeros(1,2,2)))

@pytest.mark.parametrize('shading',[False,True])
def test_actual_renderer_shape_and_background_gradient(shading):
    class Settings:
        def __init__(self,**kw):self.kw=kw
    class Raster:
        def __init__(self,raster_settings):self.s=raster_settings
        def __call__(self,**kw):
            h,w=self.s.kw['image_height'],self.s.kw['image_width']
            signal=kw['means3D'].sum()*0.001+kw['means2D'].sum()*0.001
            return torch.ones(3,h,w)*0.1+signal, torch.ones(5), torch.ones(1,h,w), torch.ones(1,h,w)*0.5
    suffix='_shading' if shading else ''
    cls,env=load_class('threestudio/models/renderers/diff_gaussian_rasterizer'+suffix+'.py','DiffGaussian',math=__import__('math'),GaussianRasterizationSettings=Settings,GaussianRasterizer=Raster)
    model=cls();model.cfg=NS(debug=False,invert_bg_prob=0.5)
    model.geometry=NS(get_xyz=torch.randn(5,3,requires_grad=True),active_sh_degree=0,get_opacity=torch.ones(5,1),get_scaling=torch.ones(5,3),get_rotation=torch.ones(5,4),get_features=torch.ones(5,1,3))
    model.normal_module=env['Depth2Normal']()
    model.material=lambda **kw:kw['albedo']
    bg=torch.full((1,3,5,3),0.4,requires_grad=True)
    camera=NS(FoVx=0.7,FoVy=0.5,image_height=3,image_width=5,world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4),camera_center=torch.zeros(3))
    out=model(camera,torch.zeros(3),batch_idx=0,rays_d=torch.ones(1,3,5,3),rays_o=torch.zeros(1,3,5,3),light_positions=torch.ones(1,3),view_background=bg)
    assert out['comp_rgb_bg'].shape==(1,3,5,3)
    assert out['render'].shape==(3,3,5)
    out['render'].sum().backward()
    assert bg.grad is not None and bg.grad.abs().sum()>0
    assert out['viewspace_points'].grad is not None

def test_actual_gaussian_system_stacks_views_and_non_square_fov():
    calls=[]
    def cam(**kw):calls.append(kw);return torch.eye(4),torch.eye(4),torch.zeros(3)
    cls,_=load_class('threestudio/systems/gaussian_mvdream.py','MVDreamSystem',get_cam_info_gaussian=cam,Camera=lambda **kw:NS(**kw),autocast=lambda **kw:__import__('contextlib').nullcontext())
    s=cls();s.global_step=0;s.background_tensor=torch.zeros(3)
    s.geometry=NS(cfg=NS(position_lr_max_steps=0,scale_lr_max_steps=0))
    s.background=lambda dirs:torch.arange(4.)[:,None,None,None].expand(4,2,3,3)
    def render(camera,bg,**kw):
        return dict(render=torch.zeros(3,2,3),comp_rgb_bg=kw['view_background'],viewspace_points=torch.zeros(5,3),visibility_filter=torch.ones(5,dtype=torch.bool),radii=torch.ones(5),depth=torch.ones(1,2,3))
    s.renderer=render
    batch=dict(c2w=torch.eye(4).repeat(4,1,1),fovy=torch.ones(4)*0.6,width=3,height=2,rays_d=torch.ones(4,2,3,3))
    out=s.forward(batch)
    assert out['comp_rgb_bg'].shape==(4,2,3,3)
    assert out['comp_rgb_bg'][:,0,0,0].tolist()==[0,1,2,3]
    assert 'batch_idx' not in batch
    assert calls[0]['fovx']>calls[0]['fovy']
    assert out['comp_depth'].shape==(4,2,3,1)

def test_actual_manual_optimizer_order_and_one_step_counter():
    events=[]
    cls,_=load_class('threestudio/systems/gaussian_mvdream.py','MVDreamSystem')
    s=cls();s.optim_num=2;s.trainer=NS(precision='32-true');s.global_step=0
    losses={k:0 for k in ['lambda_position','lambda_opacity','lambda_scales','lambda_tv_loss']};losses['lambda_sds']=1
    s.cfg=NS(image=False,refinement=False,loss=losses);s.prompt_utils=None;s.log=lambda *a,**k:None;s.C=lambda x:x
    param=torch.nn.Parameter(torch.tensor(1.))
    def counted_step():events.append('gaussian_step');s.global_step+=1
    opt=NS(zero_grad=lambda **k:None,step=counted_step)
    net=NS(zero_grad=lambda **k:None,optimizer=NS(step=lambda:events.append('background_step')))
    s.optimizers=lambda:(opt,net)
    s.forward=lambda batch:dict(visibility_filter=[],radii=[],viewspace_points=[],comp_rgb=param)
    s.guidance=lambda *a,**k:dict(loss_sds=param.square())
    s.geometry=NS(get_xyz=torch.zeros(5,3),update_states=lambda *a:events.append('densify'))
    def backward(loss):events.append('backward');loss.backward()
    s.manual_backward=backward
    s.training_step({},0)
    assert events==['backward','gaussian_step','background_step','densify']
    assert s.global_step==1 and param.grad==2

def test_actual_imagedream_reference_view_timestep_alignment():
    identity=lambda *a,**k:None
    cls,_=load_class('threestudio/models/guidance/image_multiview_diffusion_guidance.py','MultiviewDiffusionGuidance',T=NS(Compose=identity,Resize=identity,ToTensor=identity,Normalize=identity))
    model=cls();model.cfg=NS(image_size=256,n_view=4)
    context={'context':torch.arange(8.)[:,None,None],'ip':torch.ones(8,2,3),'camera':torch.ones(8,16)}
    latent,t,context=model.append_extra_view(torch.ones(8,4,2,2),torch.tensor([1]*4+[9]*4),context)
    assert latent.shape[0]==10 and t.tolist()==[1]*5+[9]*5
    assert latent[4].count_nonzero()==0 and latent[9].count_nonzero()==0
    assert context['camera'][4].count_nonzero()==0

@pytest.mark.parametrize('image',[False,True])
def test_actual_sds_guidance_backprop_without_model_weights(image):
    path='image_multiview_diffusion_guidance' if image else 'multiview_diffusion_guidance'
    cls,_=load_class('threestudio/models/guidance/'+path+'.py','MultiviewDiffusionGuidance',add_random_background=lambda image,bg:image)
    s=cls();s.cfg=NS(n_view=4,ip_mode=None,view_dependent_prompting=False,guidance_scale=2.,recon_loss=False)
    s.num_train_timesteps=10;s.min_step=1;s.max_step=8;s.grad_clip_val=None;s.alphas_cumprod=torch.linspace(.9,.1,10)
    s.get_camera_cond=lambda camera,fovy:camera.flatten(1)
    s.model=NS(eval=lambda:None,q_sample=lambda z,t,n:z+n,apply_model=lambda z,t,c:z*.2,get_learned_image_conditioning=lambda img:torch.ones(1,2,3))
    prompt=NS(image=__import__('PIL.Image',fromlist=['Image']).new('RGB',(2,2)) if image else None)
    latent=torch.randn(4,4,2,2,requires_grad=True)
    out=s.forward(latent,prompt,torch.zeros(4),torch.zeros(4),torch.ones(4),torch.eye(4).repeat(4,1,1),input_is_latent=True,text_embeddings=torch.ones(8,2,3),comp_rgb_bg=torch.ones(4,2,2,3))
    out['loss_sds'].backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum()>0

def test_finetune_loss_trains_attention_and_ignores_extra_view():
    ft=load_file('scripts/finetune_multiview.py')
    weight=torch.nn.Parameter(torch.tensor(.1))
    fake=NS(parameterization='eps',q_sample=lambda z,t,n:z+n,apply_model=lambda z,t,c:z*weight)
    loss=ft.epsilon_loss(fake,torch.ones(4,4,2,2),torch.ones(4,dtype=torch.long),{},torch.ones(4,4,2,2))
    loss.backward()
    assert weight.grad is not None and weight.grad!=0

def test_runner_requires_image_files_and_keeps_prompt_one_argument(tmp_path):
    runner=load_file('scripts/run_pipeline.py')
    args=NS(pipeline='imagedream-3dgs',prompt='a red fox; $(echo no)',image=None,checkpoint=None,model_config=None,smoke=False,overrides=[])
    with pytest.raises(ValueError):runner.build_command(args)
    for name in ['image','checkpoint','model_config']:
        f=tmp_path/name;f.write_text('fixture');setattr(args,name,str(f))
    command=runner.build_command(args)
    assert len([v for v in command if v.startswith('system.prompt_processor.prompt=')])==1

@pytest.mark.parametrize('image',[False,True])
def test_actual_guidance_configure_initializes_schedule(image):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('alphas_cumprod',torch.linspace(.99,.01,1000))
    name='image_multiview_diffusion_guidance' if image else 'multiview_diffusion_guidance'
    cls,_=load_class('threestudio/models/guidance/'+name+'.py','MultiviewDiffusionGuidance',build_model=lambda *a,**k:TinyModel(),threestudio=NS(info=lambda *a:None),C=lambda v,*a:v)
    s=cls();s.device='cpu';s.cfg=NS(model_name='dummy',config_path=None,ckpt_path=None,min_step_percent=.02,max_step_percent=.98)
    s.configure()
    assert torch.equal(s.alphas_cumprod,s.model.alphas_cumprod)
    assert s.min_step==20 and s.max_step==980
    assert 'alphas_cumprod' in dict(s.named_buffers())

def test_actual_legacy_checkpoint_stats_added_and_preserved():
    cls,_=load_class('threestudio/systems/gaussian_mvdream.py','MVDreamSystem',BasicPointCloud=lambda **kw:NS(**kw))
    s=cls();s.geometry=NS(create_from_pcd=lambda *a:None,training_setup=lambda:None,max_radii2D=torch.zeros(3),xyz_gradient_accum=torch.zeros(3,1),denom=torch.zeros(3,1))
    ckpt={'state_dict':{'geometry._xyz':torch.ones(3,3),'geometry.denom':torch.ones(3,1)*7}}
    s.on_load_checkpoint(ckpt)
    assert ckpt['state_dict']['geometry.max_radii2D'].shape==(3,)
    assert ckpt['state_dict']['geometry.denom'].sum()==21
    tree=ast.parse((ROOT/'threestudio/models/geometry/gaussian_base.py').read_text())
    buffers={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='register_buffer' and n.args and isinstance(n.args[0],ast.Constant)}
    assert {'max_radii2D','xyz_gradient_accum','denom'} <= buffers

def test_finetune_manifest_rejects_invalid_camera(tmp_path):
    import json
    ft=load_file('scripts/finetune_multiview.py')
    record={'prompt':'object','reference':'ref.png','views':[{'image':'view.png','c2w':[[0]*4]*4} for _ in range(4)]}
    manifest=tmp_path/'data.jsonl';manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='homogeneous'):ft.read_records(manifest)

def test_multi_view_config_connections():
    import yaml
    for name in ('mvdream','imagedream'):
        for representation in ('nerf','3dgs'):
            cfg=yaml.safe_load((ROOT/f'configs/{name}-{representation}-coarse.yaml').read_text())
            n=cfg['data']['n_view'];sizes=cfg['data']['batch_size']
            assert all(x%n==0 for x in (sizes if isinstance(sizes,list) else [sizes]))
            if name=='imagedream':
                assert cfg['system']['image'] is True
                assert cfg['system']['prompt_processor']['image_path']=='???'
                assert cfg['system']['guidance']['model_name']=='sd-v2.1-base-4view-ipmv'

def test_actual_camera_projection_preserves_cpu_device():
    source=ast.parse((ROOT/'threestudio/utils/ops.py').read_text())
    names={'convert_pose','get_projection_matrix_gaussian','get_cam_info_gaussian'}
    tree=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    env={'torch':torch,'math':__import__('math')}
    exec(compile(ast.fix_missing_locations(tree),'camera-functions','exec'),env)
    w2c,proj,center=env['get_cam_info_gaussian'](torch.eye(4),torch.tensor(.8),torch.tensor(.6),.1,100)
    assert proj.device.type=='cpu' and w2c.shape==proj.shape==(4,4) and center.shape==(3,)
    assert torch.isfinite(proj).all()
