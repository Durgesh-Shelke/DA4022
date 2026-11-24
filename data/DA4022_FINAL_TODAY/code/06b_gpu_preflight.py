# This gate runs BEFORE donor training and the long experiments. The random
# placeholder bases below are ONLY for testing execution, not learned priors.
preflight_record=records_for('validation')[0]
RUNNER.clear()
placeholder_bank={}
for i,name in enumerate(RUNNER.names):
    base=RUNNER.model.get_submodule(name)
    generator=torch.Generator().manual_seed(700+i)
    placeholder_bank[name]={
        'U':torch.linalg.qr(torch.randn(base.out_features,CFG['basis_dim'],generator=generator),mode='reduced').Q,
        'V':torch.linalg.qr(torch.randn(base.in_features,CFG['basis_dim'],generator=generator),mode='reduced').Q,
    }

preflight_folders={}
for method in ['lora4','lora1','learned','random','hybrid','shared_real','learned_energy']:
    uses_bank=method not in ['lora4','lora1','random']
    folder=experiment('gpu_preflight',preflight_record,method,
        .01 if uses_bank or method=='random' else .001,
        steps=2,bank=placeholder_bank if uses_bank else None,
        bank_name='preflight_random_placeholder' if uses_bank else None)
    result=torch.load(folder/'result.pt',map_location='cpu',weights_only=False)
    assert result['history'][-1]['updates']==2, 'Two successful updates not completed'
    assert all(torch.isfinite(t).all() for layer in result['adapter'].values() for t in layer.values()), 'Invalid adapter weights'
    # Every adapted layer should receive at least some gradient signal.
    active=sum(float(layer.get('B',torch.zeros(1)).abs().sum())+
               float(layer.get('M',torch.zeros(1)).abs().sum()) for layer in result['adapter'].values())
    assert active>0, 'Optimizer did not change any zero-initialized correction'
    assert np.isfinite(result['history'][-1]['unobserved_absrel_pct'])
    preflight_folders[method]=folder

# Verify that the saved adapter/optimizer/scaler checkpoint can actually be
# restored into fresh objects, and that it reproduces the final prediction.
folder=preflight_folders['lora4']
saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
finished=torch.load(folder/'result.pt',map_location='cpu',weights_only=False)
RUNNER.install('lora4',47)
RUNNER.restore_adapter(saved['adapter'])
test_optimizer=torch.optim.AdamW(RUNNER.groups(.001),weight_decay=0.)
test_optimizer.load_state_dict(saved['optimizer'])
test_scaler=torch.amp.GradScaler('cuda',enabled=RUNNER.dtype==torch.float16)
test_scaler.load_state_dict(saved['scaler'])
image,truth,valid,condition,mask,_=make_condition(preflight_record,100,0.)
with torch.no_grad():
    processed=RUNNER.preprocess(image)
    raw=RUNNER.predict(processed,tuple(truth.shape[-2:]))
    restored,_,_=RUNNER.align(raw,condition.to(RUNNER.device),mask.to(RUNNER.device))
assert torch.allclose(restored.cpu(),finished['prediction'],atol=2e-3,rtol=2e-3), 'Checkpoint prediction round-trip failed'
test_optimizer=None;test_scaler=None
del processed,raw,restored,result,saved,finished,placeholder_bank
RUNNER.clear();gc.collect();torch.cuda.empty_cache()
write_json(ROOT/'GPU_PREFLIGHT_PASSED.json',dict(time=utcnow(),methods=list(preflight_folders),
    updates_per_method=2,checkpoint_roundtrip=True,capa_commit=CAPA_COMMIT,config_hash=CONFIG_HASH))
journal('gpu_preflight_passed',methods=list(preflight_folders))
print('\nGPU PREFLIGHT PASSED: all seven methods completed forward/backward/update, finite metrics, saving and checkpoint restoration.')
print('The full experiment cells may now run. This is execution validation, not evidence of model quality.')
backup()
