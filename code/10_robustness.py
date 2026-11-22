# Sparse-point and noise studies use the FIRST selected frame from EACH test
# scene, a common 30-update budget and fixed settings. No test retuning.
def first_test_frames():
    return [next(r for r in records_for('test') if r['scene']==scene) for scene in CFG['split']['test']]

def run_robustness():
    conditions=[(100,0.),(20,0.),(500,0.),(100,.1),(100,.2)]
    for points,noise in conditions:
        for rec in first_test_frames():
            experiment('robustness',rec,'frozen',0.,0,points=points,noise=noise)
        for method in ['lora4','learned','random','hybrid']:
            bank,label=bank_for(method)
            for rec in first_test_frames():
                experiment('robustness',rec,method,SELECTED[method],CFG['robustness_steps'],
                    points=points,noise=noise,bank=bank,bank_name=label)

if CFG['run_robustness'] and STAGE_STATUS.get('main')=='complete':
    try:
        run_robustness();STAGE_STATUS['robustness']='complete'
    except BudgetStop as exc:
        STAGE_STATUS['robustness']='budget_stopped';print(exc)
    except Exception:
        STAGE_STATUS['robustness']='failed';backup();raise
else:print('Robustness skipped: disabled or main stage incomplete.')
write_json(ROOT/'stage_status.json',STAGE_STATUS)
backup()
