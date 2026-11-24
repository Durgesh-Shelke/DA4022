SELECTED={}
METHODS=['lora4','lora1','learned','random','hybrid','shared_real','learned_energy']

def bank_for(method):
    if method=='learned_energy':return BANKS['combined_energy'],'combined_energy'
    if method=='shared_real':return BANKS['shared_real'],'shared_real'
    if method in ['learned','hybrid']:return BANKS['combined'],'combined'
    return None,None

def final_row(folder):
    return pd.read_csv(Path(folder)/'history.csv').sort_values('updates').iloc[-1]

def validate():
    assert all(k in BANKS for k in ['combined','shared_real']), 'Finish donor/basis stage first.'
    frozen=ROOT/'selected_settings.json'
    if frozen.exists():
        payload=json.loads(frozen.read_text())
        assert payload['config_hash']==CONFIG_HASH
        SELECTED.update(payload['selected'])
        print('Reusing frozen validation selections:',SELECTED)
        return
    candidates=[]
    for method in METHODS:
        bank,label=bank_for(method)
        for lr in CFG['lr_candidates'][method]:
            values=[]
            for rec in records_for('validation'):
                p=experiment('validation',rec,method,lr,CFG['validation_steps'],
                             bank=bank,bank_name=label)
                values.append(float(final_row(p)['unobserved_absrel_pct']))
            candidates.append(dict(method=method,lr=lr,mean_validation_absrel_pct=float(np.mean(values)),frames=len(values)))
    table=pd.DataFrame(candidates)
    table.to_csv(ROOT/'validation_candidates.csv',index=False)
    for method in METHODS:
        winner=table[table.method==method].sort_values(['mean_validation_absrel_pct','lr']).iloc[0]
        SELECTED[method]=float(winner.lr)
    write_json(frozen,dict(config_hash=CONFIG_HASH,selected=SELECTED,
        selection='mean final unobserved AbsRel on validation frames only; all target evaluation settings frozen afterward',
        validation_steps=CFG['validation_steps'],test_steps=CFG['test_steps']))
    journal('validation_settings_frozen',selected=SELECTED)
    print(table.to_string(index=False));print('FROZEN SETTINGS:',SELECTED)

if STAGE_STATUS.get('donors')=='complete':
    try:
        validate();STAGE_STATUS['validation']='complete'
    except BudgetStop as exc:
        STAGE_STATUS['validation']='budget_stopped';print(exc)
    except Exception:
        STAGE_STATUS['validation']='failed';backup();raise
else:print('Validation not run: donor stage incomplete.')
write_json(ROOT/'stage_status.json',STAGE_STATUS)
backup()
