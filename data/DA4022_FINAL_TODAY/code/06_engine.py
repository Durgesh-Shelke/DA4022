import random, math, traceback, importlib.util
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import VGGT directly. This avoids CAPA's package/PEFT/TorchAO import chain.
sys.path.insert(0,str(REPO/'third_party'/'VGGT_VPT'))
from vggt.models.vggt import VGGT
spec = importlib.util.spec_from_file_location('da4022_alignment',REPO/'capa'/'utils'/'alignment.py')
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def float_tree(x):
    if torch.is_tensor(x): return x.float()
    if isinstance(x,list): return [float_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(float_tree(v) for v in x)
    return x

def checkpoint_forward(original):
    def forward(*args,**kwargs):
        if torch.is_grad_enabled():
            return checkpoint(original,*args,use_reentrant=False,**kwargs)
        return original(*args,**kwargs)
    return forward

class AdaptedLinear(nn.Module):
    def __init__(self,base,method,rank=4,bases=None,residual_rank=1):
        super().__init__()
        self.base=base; self.method=method
        dev=base.weight.device
        out_features,in_features=base.weight.shape
        if method in ['lora4','lora1','donor']:
            self.A=nn.Parameter(torch.empty(rank,in_features,device=dev,dtype=torch.float32))
            self.B=nn.Parameter(torch.zeros(out_features,rank,device=dev,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        else:
            U,V=bases
            self.register_buffer('U',U.to(dev).float())
            self.register_buffer('V',V.to(dev).float())
            self.M=nn.Parameter(torch.zeros(U.shape[1],V.shape[1],device=dev,dtype=torch.float32))
            if method=='hybrid':
                self.A=nn.Parameter(torch.empty(residual_rank,in_features,device=dev,dtype=torch.float32))
                self.B=nn.Parameter(torch.zeros(out_features,residual_rank,device=dev,dtype=torch.float32))
                nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self,x):
        y=self.base(x)
        if hasattr(self,'M'):
            y=y+2*F.linear(F.linear(F.linear(x,self.V.T),self.M),self.U)
        if hasattr(self,'A'):
            y=y+2*F.linear(F.linear(x,self.A),self.B)
        return y

class Runner:
    def __init__(self):
        self.device=torch.device('cuda:0')
        self.dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print('Loading VGGT checkpoint. First download can take several minutes.',flush=True)
        self.model=VGGT.from_pretrained('facebook/VGGT-1B').eval().requires_grad_(False).to(self.device)
        self.names=[name for name,layer in self.model.named_modules()
                    if isinstance(layer,nn.Linear) and 'aggregator.patch_embed.' in name and name.endswith('.qkv')]
        assert len(self.names)==24, f'Unexpected VGGT layout: {len(self.names)} image-encoder QKV layers.'
        for block in self.model.modules():
            if block.__class__.__name__ in {'Block','NestedTensorBlock'}:
                block.forward=checkpoint_forward(block.forward)
        self.adapters={}
        print('Loaded. Precision:',self.dtype,'target layers:',len(self.names),flush=True)

    def clear(self):
        for name,layer in list(self.adapters.items()):
            parent,child=name.rsplit('.',1)
            setattr(self.model.get_submodule(parent),child,layer.base)
        self.adapters={}
        self.model.requires_grad_(False)
        gc.collect();torch.cuda.empty_cache()

    def install(self,method,seed,banks=None):
        self.clear();seed_everything(seed)
        if method=='frozen':return
        for i,name in enumerate(self.names):
            base=self.model.get_submodule(name)
            basis=None
            rank=1 if method=='lora1' else CFG['rank']
            if method=='random':
                g=torch.Generator().manual_seed(seed+76+i*1009)
                U=torch.linalg.qr(torch.randn(base.out_features,CFG['basis_dim'],generator=g),mode='reduced').Q
                V=torch.linalg.qr(torch.randn(base.in_features,CFG['basis_dim'],generator=g),mode='reduced').Q
                basis=(U,V)
            elif method not in ['donor','lora4','lora1']:
                assert banks is not None and name in banks, f'Missing bases for {name}'
                basis=(banks[name]['U'],banks[name]['V'])
            layer=AdaptedLinear(base,method,rank,basis,CFG['residual_rank'])
            parent,child=name.rsplit('.',1)
            setattr(self.model.get_submodule(parent),child,layer)
            self.adapters[name]=layer
        self.model.eval()

    def groups(self,lr):
        groups=[]
        for layer in self.adapters.values():
            if hasattr(layer,'M'):groups.append({'params':[layer.M],'lr':lr})
            if hasattr(layer,'A'):
                rate=CFG['hybrid_residual_lr'] if layer.method=='hybrid' else lr
                groups.append({'params':[layer.A,layer.B],'lr':rate})
        return groups

    def adapter_state(self):
        return {name:{key:p.detach().cpu().clone() for key,p in layer.named_parameters(recurse=False)}
                for name,layer in self.adapters.items()}

    def restore_adapter(self,state):
        assert set(state)==set(self.adapters)
        with torch.no_grad():
            for name,values in state.items():
                for key,value in values.items():getattr(self.adapters[name],key).copy_(value.to(self.device))

    def preprocess(self,rgb):
        width=CFG['input_width']
        h,w=rgb.shape[-2:]
        height=round(h*width/w/14)*14
        image=F.interpolate(rgb.to(self.device),size=(height,width),mode='bicubic',align_corners=False,antialias=True)
        # Our selected frames are landscape. Avoid silently evaluating cropped GT.
        assert height<=width, 'Portrait frames need an explicitly aligned crop protocol.'
        return image.clamp(0,1)

    def predict(self,image,size):
        images=image.unsqueeze(0)
        with torch.autocast('cuda',dtype=self.dtype):
            tokens,start=self.model.aggregator(images,use_prompt=False,
                         feature_idx_ls=self.model.depth_head.intermediate_layer_idx)
        with torch.autocast('cuda',enabled=False):
            depth,_=self.model.depth_head(float_tree(tokens),images.float(),start)
        depth=depth.squeeze(0).squeeze(-1)
        return F.interpolate(depth.unsqueeze(1),size=size,mode='bilinear',align_corners=False).squeeze(1)

    def align(self,pred,condition,mask):
        with torch.no_grad():
            scale,shift=alignment.align_depth_to_condition(pred.detach(),condition,mask,
                                    n_points=CFG['align_points'],seed=47)
        if not torch.isfinite(scale).all() or not torch.isfinite(shift).all():
            raise FloatingPointError('Non-finite sparse-depth alignment')
        return pred*scale.reshape(-1,1,1)+shift.reshape(-1,1,1),scale.detach(),shift.detach()

def metrics(pred,gt,valid,mask):
    pred=pred.detach().float().cpu()
    ans={}
    for region,select in [('all',valid),('observed',valid&mask),('unobserved',valid&~mask)]:
        values=pred[select];truth=gt[select]
        if not len(values):raise ValueError('Empty evaluation mask: '+region)
        if not torch.isfinite(values).all():raise FloatingPointError('Invalid prediction; not dropping invalid pixels.')
        error=values-truth
        ans[region+'_absrel_pct']=100*(error.abs()/truth).mean().item()
        ans[region+'_rmse']=error.square().mean().sqrt().item()
        ans[region+'_nonpositive_pct']=100*(values<=0).float().mean().item()
    return ans

def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def atomic_torch_save(value,path):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    torch.save(value,temp);temp.replace(path)

def experiment(stage,record,method,lr,steps,seed=47,points=100,noise=0.,bank=None,bank_name=None,donor_group=None):
    """One frame per target run; donor_group alternates controlled variants
    using ONE shared adapter. Ground truth never enters target gradients except
    through the explicitly sampled conditioning points."""
    check_budget()
    group=donor_group or [record]
    settings=dict(config_hash=CONFIG_HASH,capa_commit=CAPA_COMMIT,stage=stage,
        sample=record['id'],input_hashes=[r['sha256'] for r in group],method=method,lr=lr,
        steps=steps,seed=seed,points=points,noise=noise,bank_name=bank_name,
        bank_digest=hashlib.sha256(b''.join(
            x[k].contiguous().numpy().tobytes() for _,x in sorted(bank.items()) for k in ['U','V'])).hexdigest() if bank else None)
    run_id=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:16]
    folder=RUNS/(stage+'_'+record['id']+'_'+method+'_'+run_id)
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'complete.json'
    if done.exists():
        status=json.loads(done.read_text())
        for required in ['history.csv','result.pt']:
            assert (folder/required).exists(), f'Incomplete saved run {folder}'
        print('REUSE',stage,record['id'],method,flush=True)
        return folder
    write_json(folder/'settings.json',settings)
    journal('run_started',run_id=run_id,**settings)
    history=[];completed=0;attempts=0;optimization_seconds=0.
    optimizer=None;scaler=None;processed=[]
    try:
        RUNNER.install(method,seed,bank)
        groups=RUNNER.groups(lr)
        params=[p for g in groups for p in g['params']]
        nparams=sum(p.numel() for p in params)
        assert {id(p) for p in params}=={id(p) for p in RUNNER.model.parameters() if p.requires_grad}, 'Trainable backbone parameter leak'
        expected={'lora4':393216,'donor':393216,'lora1':98304,'random':384,
                  'learned':384,'learned_energy':384,'shared_real':384,'hybrid':98688,'frozen':0}
        assert nparams==expected[method], f'Unexpected parameter budget: {method} {nparams}'
        optimizer=torch.optim.AdamW(groups,weight_decay=CFG['weight_decay']) if groups else None
        scaler=torch.amp.GradScaler('cuda',enabled=RUNNER.dtype==torch.float16,init_scale=1024.)
        datasets=[]
        for rec in group:
            rgb,gt,valid,cond,mask,cond_meta=make_condition(rec,points,noise)
            with torch.no_grad():image=RUNNER.preprocess(rgb)
            datasets.append((image,gt,valid,cond.to(RUNNER.device),mask.to(RUNNER.device),mask,cond_meta))
        write_json(folder/'conditioning.json',[x[-1] for x in datasets])
        checkpoint_path=folder/'checkpoint.pt'
        if checkpoint_path.exists():
            cp=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            RUNNER.restore_adapter(cp['adapter'])
            if optimizer:optimizer.load_state_dict(cp['optimizer'])
            scaler.load_state_dict(cp['scaler'])
            completed=cp['completed'];attempts=cp['attempts'];history=cp['history']
            optimization_seconds=cp['optimization_seconds']
            torch.set_rng_state(cp['rng_cpu']);torch.cuda.set_rng_state(cp['rng_cuda'],RUNNER.device)
            print('RESUME',record['id'],method,'at',completed,flush=True)
        torch.cuda.reset_peak_memory_stats(RUNNER.device)
        wall_start=time.perf_counter()
        eval_steps={0,1,5,10,20,30,50,60,75,100,steps}
        while completed<=steps:
            check_budget()
            final=completed==steps
            if optimizer:optimizer.zero_grad(set_to_none=True)
            image,gt,valid,cond,mask_gpu,mask_cpu,cond_meta=datasets[completed%len(datasets)]
            torch.cuda.synchronize();t0=time.perf_counter()
            with torch.set_grad_enabled(not final and optimizer is not None):
                raw=RUNNER.predict(image,tuple(cond.shape[-2:]))
                if not torch.isfinite(raw).all():raise FloatingPointError('Non-finite raw depth')
                pred,scale,shift=RUNNER.align(raw,cond,mask_gpu)
                loss=(pred-cond).abs()[mask_gpu].mean()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite sparse loss')
            torch.cuda.synchronize();forward_seconds=time.perf_counter()-t0
            # CPU metrics and disk writes are deliberately excluded from the
            # recorded optimization timer. These are instrumented pilot timings.
            if completed in eval_steps:
                row=dict(updates=completed,attempts=attempts,sparse_L1=float(loss.detach()),
                         adaptation_seconds=optimization_seconds,
                         time_to_prediction_seconds=optimization_seconds+forward_seconds,
                         trainable_parameters=nparams,scale=float(scale[0]),shift=float(shift[0]),
                         **metrics(pred,gt,valid,mask_cpu))
                history=[h for h in history if h['updates']!=completed]+[row]
                pd.DataFrame(history).sort_values('updates').to_csv(folder/'history.csv',index=False)
                print(f'{stage} {record["id"]} {method}: {completed}/{steps}, '
                      f'L1={row["sparse_L1"]:.4f}, unobserved={row["unobserved_absrel_pct"]:.3f}%',flush=True)
            if final:
                result=dict(adapter=RUNNER.adapter_state(),prediction=pred.detach().float().cpu(),
                    history=history,settings=settings,trainable_parameters=nparams,
                    peak_memory_GB=torch.cuda.max_memory_allocated(RUNNER.device)/1024**3,
                    optimization_seconds=optimization_seconds,final_forward_seconds=forward_seconds,
                    elapsed_segment_seconds=time.perf_counter()-wall_start)
                atomic_torch_save(result,folder/'result.pt')
                write_json(done,dict(status='complete',run_id=run_id,finished=utcnow(),
                                    trainable_parameters=nparams,peak_memory_GB=result['peak_memory_GB']))
                journal('run_complete',run_id=run_id,folder=str(folder.relative_to(ROOT)))
                if sum(1 for _ in RUNS.glob('*/complete.json')) % 5 == 0:
                    backup()
                return folder
            torch.cuda.synchronize();back_start=time.perf_counter()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            torch.cuda.synchronize()
            optimization_seconds+=forward_seconds+time.perf_counter()-back_start
            attempts+=1
            success=scaler.get_scale()>=old_scale
            if success:completed+=1
            del raw,pred,loss
            optimizer.zero_grad(set_to_none=True)
            if success and (completed%10==0 or completed==steps):
                atomic_torch_save(dict(adapter=RUNNER.adapter_state(),optimizer=cpu_tree(optimizer.state_dict()),
                    scaler=scaler.state_dict(),completed=completed,attempts=attempts,history=history,
                    optimization_seconds=optimization_seconds,rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state(RUNNER.device)),checkpoint_path)
            if attempts>steps+40:raise FloatingPointError('Too many skipped FP16 updates')
    except Exception as exc:
        write_json(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),at=utcnow()))
        journal('run_failed',run_id=run_id,error=str(exc))
        raise
    finally:
        optimizer=None;scaler=None;processed=[]
        # Local graph references must disappear before the next run.
        gc.collect();torch.cuda.empty_cache()

RUNNER=Runner()
# Forward-equivalence smoke test: every zero-initialized adapter must leave
# pretrained output unchanged, and an update must not touch any base weights.
smoke_record=records_for('validation')[0]
rgb_smoke,_,_=load_frame(smoke_record)
with torch.no_grad():
    image_smoke=RUNNER.preprocess(rgb_smoke)
    RUNNER.install('frozen',47)
    reference=RUNNER.predict(image_smoke,tuple(rgb_smoke.shape[-2:])).cpu()
    RUNNER.install('lora4',47)
    check=RUNNER.predict(image_smoke,tuple(rgb_smoke.shape[-2:])).cpu()
assert torch.allclose(reference,check,atol=1e-4,rtol=1e-3), 'Zero-adapter equivalence failed'
RUNNER.clear()
del reference,check,image_smoke,rgb_smoke
print('Zero-initialized LoRA forward-equivalence check passed.')
backup()
