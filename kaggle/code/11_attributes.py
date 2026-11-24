# Diagnose individual synthetic attribute bases on VALIDATION ONLY. This does
# not retune or select the combined basis using test outcomes. Basis dimensions,
# coefficient LR and update budget are identical across these ablations.
def attribute_ablation():
    rows=[]
    for label in CFG['synthetic_attributes']+['combined']:
        for rec in records_for('validation'):
            p=experiment('attribute_validation',rec,'learned',SELECTED['learned'],CFG['validation_steps'],
                         bank=BANKS[label],bank_name=label)
            row=final_row(p)
            rows.append(dict(attribute_basis=label,scene=rec['scene'],sample=rec['id'],
                unobserved_absrel_pct=float(row['unobserved_absrel_pct']),
                sparse_L1=float(row['sparse_L1'])))
    table=pd.DataFrame(rows);table.to_csv(ROOT/'attribute_validation.csv',index=False)
    print(table.groupby('attribute_basis').unobserved_absrel_pct.mean())
    # Descriptive row/column basis overlaps; not a proof of semantic disentanglement.
    overlaps=[]
    attributes=CFG['synthetic_attributes']
    for a in attributes:
        for b in attributes:
            lu=[];rv=[]
            for name in RUNNER.names:
                lu.append(float((BANKS[a][name]['U'].T@BANKS[b][name]['U']).square().sum()/CFG['basis_dim']))
                rv.append(float((BANKS[a][name]['V'].T@BANKS[b][name]['V']).square().sum()/CFG['basis_dim']))
            overlaps.append(dict(attribute_a=a,attribute_b=b,left_overlap=np.mean(lu),right_overlap=np.mean(rv)))
    pd.DataFrame(overlaps).to_csv(ROOT/'attribute_basis_overlap.csv',index=False)

if CFG['run_appearance_ablation'] and STAGE_STATUS.get('validation')=='complete':
    try:
        attribute_ablation();STAGE_STATUS['attribute_ablation']='complete'
    except BudgetStop as exc:
        STAGE_STATUS['attribute_ablation']='budget_stopped';print(exc)
    except Exception:
        STAGE_STATUS['attribute_ablation']='failed';backup();raise
else:print('Attribute ablation skipped: disabled or validation incomplete.')
write_json(ROOT/'stage_status.json',STAGE_STATUS)
backup()
