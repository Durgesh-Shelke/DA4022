SESSION_START=time.monotonic()  # GPU-stage wall-time budget starts after data/model loading.
STAGE_STATUS={}
BANKS={}

def load_result(folder):
    return torch.load(Path(folder)/'result.pt',map_location='cpu',weights_only=False)

def extract_energy_basis(folders,label):
    """Top left/right uncentered update-energy subspaces. Each update is
    Frobenius-normalized. This is a simple pooling baseline, not a replication
    of Jiang et al.'s robust shared-mean solver."""
    states=[load_result(p)['adapter'] for p in folders]
    assert states
    bank={};diagnostics=[]
    for name in RUNNER.names:
        left,right=[],[]
        for state in states:
            A,B=state[name]['A'].float(),state[name]['B'].float()
            qB,rB=torch.linalg.qr(B,mode='reduced')
            qA,rA=torch.linalg.qr(A.T,mode='reduced')
            u,s,vh=torch.linalg.svd(rB@rA.T,full_matrices=False)
            assert torch.isfinite(s).all() and s.norm()>1e-10, 'Empty/invalid donor update'
            # Normalize each effective donor update, avoiding one large adapter
            # dominating solely because of its displacement magnitude.
            normalized=s/s.norm()
            left.append((qB@u)*normalized)
            right.append((qA@vh.T)*normalized)
        u,sl,_=torch.linalg.svd(torch.cat(left,dim=1),full_matrices=False)
        v,sr,_=torch.linalg.svd(torch.cat(right,dim=1),full_matrices=False)
        d=CFG['basis_dim']
        assert min(len(sl),len(sr))>=d
        U,V=u[:,:d].contiguous(),v[:,:d].contiguous()
        assert torch.allclose(U.T@U,torch.eye(d),atol=2e-5)
        assert torch.allclose(V.T@V,torch.eye(d),atol=2e-5)
        bank[name]={'U':U,'V':V}
        diagnostics.append(dict(bank=label,layer=name,left_retained_energy=float(sl[:d].square().sum()/sl.square().sum()),
            right_retained_energy=float(sr[:d].square().sum()/sr.square().sum())))
    folder=ROOT/'bases';folder.mkdir(exist_ok=True)
    atomic_torch_save(bank,folder/(label+'.pt'))
    pd.DataFrame(diagnostics).to_csv(folder/(label+'_energy.csv'),index=False)
    write_json(folder/(label+'_sources.json'),dict(label=label,source_runs=[str(p.relative_to(ROOT)) for p in folders],
        extraction='Frobenius-normalized update energy, independently pooled left and right singular subspaces',basis_dim=CFG['basis_dim']))
    return bank

def extract_robust_basis(folders,label):
    """IRLS rank-d shared mean of effective updates, using low-rank factors.
    The robust p=1 objective is approximated for a fixed number of iterations.
    Store orthonormal singular bases; only target coefficients are optimized."""
    states=[load_result(p)['adapter'] for p in folders]
    bank={};diagnostics=[]
    for name in RUNNER.names:
        pairs=[(s[name]['A'].float(),2*s[name]['B'].float()) for s in states]
        norms=torch.tensor([float(((B.T@B)*(A@A.T).T).sum()) for A,B in pairs]).clamp_min(0)
        eps=max(float(norms.sqrt().median())*1e-3,1e-8)
        weights=torch.ones(len(pairs))/len(pairs)
        for iteration in range(CFG['irls_iterations']):
            weights=weights/weights.sum()
            L=torch.cat([B*weights[i].sqrt() for i,(A,B) in enumerate(pairs)],dim=1)
            R=torch.cat([A.T*weights[i].sqrt() for i,(A,B) in enumerate(pairs)],dim=1)
            qL,rL=torch.linalg.qr(L,mode='reduced')
            qR,rR=torch.linalg.qr(R,mode='reduced')
            u,sv,vh=torch.linalg.svd(rL@rR.T,full_matrices=False)
            d=CFG['basis_dim'];s=sv[:d];U=qL@u[:,:d];V=qR@vh[:d].T
            residuals=[]
            for i,(A,B) in enumerate(pairs):
                cross=((U.T@B)@(A@V)).diagonal().mul(s).sum()
                residuals.append((norms[i]+s.square().sum()-2*cross).clamp_min(0))
            residuals=torch.stack(residuals)
            if iteration+1<CFG['irls_iterations']:
                weights=1/torch.sqrt(residuals+eps**2)
        assert torch.isfinite(U).all() and torch.isfinite(V).all()
        assert torch.allclose(U.T@U,torch.eye(d),atol=2e-5)
        assert torch.allclose(V.T@V,torch.eye(d),atol=2e-5)
        bank[name]={'U':U.contiguous(),'V':V.contiguous()}
        for i in range(len(pairs)):
            diagnostics.append(dict(bank=label,layer=name,donor_index=i,weight=float(weights[i]/weights.sum()),
                                    residual_norm=float(residuals[i].sqrt())))
    folder=ROOT/'bases';folder.mkdir(exist_ok=True)
    atomic_torch_save(bank,folder/(label+'.pt'))
    pd.DataFrame(diagnostics).to_csv(folder/(label+'_irls.csv'),index=False)
    write_json(folder/(label+'_sources.json'),dict(label=label,source_runs=[str(p.relative_to(ROOT)) for p in folders],
        extraction='Rank-d weighted shared update mean with p=1 IRLS; orthonormal left/right singular bases',
        iterations=CFG['irls_iterations'],basis_dim=CFG['basis_dim']))
    return bank

def train_donors_and_bases():
    synthetic={attribute:[] for attribute in CFG['synthetic_attributes']}
    for task in SYNTH_TASKS:
        check_budget()
        folder=experiment('synthetic_donor',task['frames'][0],'donor',CFG['donor_lr'],
            CFG['donor_steps'],points=500,donor_group=task['frames'])
        synthetic[task['attribute']].append(folder)
    for attribute,folders in synthetic.items():
        BANKS[attribute]=extract_robust_basis(folders,attribute)
    all_synthetic=[p for folders in synthetic.values() for p in folders]
    BANKS['combined']=extract_robust_basis(all_synthetic,'combined')
    BANKS['combined_energy']=extract_energy_basis(all_synthetic,'combined_energy')
    real_folders=[]
    for scene in CFG['split']['donor']:
        group=[r for r in records_for('donor') if r['scene']==scene]
        real_folders.append(experiment('real_donor',group[0],'donor',CFG['donor_lr'],
            CFG['donor_steps'],points=100,donor_group=group))
    BANKS['shared_real']=extract_robust_basis(real_folders,'shared_real')
    # Fixed/hybrid zero-output checks, without using target GT.
    rec=records_for('validation')[0]
    rgb,_,_=load_frame(rec)
    with torch.no_grad():
        image=RUNNER.preprocess(rgb)
        RUNNER.install('frozen',47);reference=RUNNER.predict(image,tuple(rgb.shape[-2:])).cpu()
        for method in ['learned','hybrid','random']:
            RUNNER.install(method,47,BANKS['combined'] if method!='random' else None)
            output=RUNNER.predict(image,tuple(rgb.shape[-2:])).cpu()
            assert torch.allclose(reference,output,atol=1e-4,rtol=1e-3), 'Zero-output invariant failed: '+method
    RUNNER.clear()
    STAGE_STATUS['donors']='complete'
    journal('bases_ready',banks=list(BANKS))

try:
    train_donors_and_bases()
except BudgetStop as exc:
    STAGE_STATUS['donors']='budget_stopped';print(exc)
except Exception:
    STAGE_STATUS['donors']='failed';backup();raise
finally:
    write_json(ROOT/'stage_status.json',STAGE_STATUS)
backup()
