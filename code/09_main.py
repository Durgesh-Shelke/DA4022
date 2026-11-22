def run_main():
    assert SELECTED,'Freeze validation choices first.'
    for rec in records_for('test'):
        # Establish a common aligned base prediction for this measurement set.
        experiment('test',rec,'frozen',0.,0)
    # Run one complete seed across all methods/scenes before extra seeds.
    # If the budget stops, the report states exactly which runs are missing.
    for seed in CFG['main_seeds']:
        for method in METHODS:
            bank,label=bank_for(method)
            for rec in records_for('test'):
                experiment('test',rec,method,SELECTED[method],CFG['test_steps'],seed=seed,
                           bank=bank,bank_name=label)

if STAGE_STATUS.get('validation')=='complete':
    try:
        run_main();STAGE_STATUS['main']='complete'
    except BudgetStop as exc:
        STAGE_STATUS['main']='budget_stopped';print(exc)
    except Exception:
        STAGE_STATUS['main']='failed';backup();raise
else:print('Main test not run: validation incomplete.')
write_json(ROOT/'stage_status.json',STAGE_STATUS)
backup()
