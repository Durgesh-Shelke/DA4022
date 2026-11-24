# Always run this cell, including after a compute-budget stop. Only completed
# runs enter result tables; missing/failed runs are listed rather than invented.
REPORT=ROOT/'report'
REPORT.mkdir(exist_ok=True)
record_lookup={r['id']:r for r in MANIFEST['frames']} if 'MANIFEST' in globals() else {}
all_rows=[];failed=[]
for folder in sorted(RUNS.iterdir()):
    if not folder.is_dir() or not (folder/'settings.json').exists():continue
    settings=json.loads((folder/'settings.json').read_text())
    if settings['config_hash']!=CONFIG_HASH:continue
    if not (folder/'complete.json').exists():
        failed.append(dict(folder=folder.name,status='failed' if (folder/'failure.json').exists() else 'incomplete'))
        continue
    status=json.loads((folder/'complete.json').read_text())
    h=pd.read_csv(folder/'history.csv').drop_duplicates('updates',keep='last')
    rec=record_lookup.get(settings['sample'],{})
    for _,row in h.iterrows():
        all_rows.append(dict(**row.to_dict(),stage=settings['stage'],sample=settings['sample'],
            scene=rec.get('scene',settings['sample']),method=settings['method'],lr=settings['lr'],
            seed=settings['seed'],points=settings['points'],noise=settings['noise'],
            bank_name=settings['bank_name'],folder=str(folder.relative_to(ROOT)),
            peak_memory_GB=status.get('peak_memory_GB',np.nan)))
ALL=pd.DataFrame(all_rows)
ALL.to_csv(REPORT/'all_completed_measurements.csv',index=False)
write_json(REPORT/'incomplete_runs.json',failed)
print('Completed runs:',len(list(RUNS.glob('*/complete.json'))),'incomplete/failed:',len(failed))

table_text=[]
if len(ALL):
    FINALS=ALL.sort_values('updates').groupby('folder',as_index=False).tail(1)
    FINALS.to_csv(REPORT/'final_results.csv',index=False)
    main=FINALS[FINALS.stage=='test'].copy()
    if len(main):
        expected_main=len(records_for('test'))*(1+len(METHODS)*len(CFG['main_seeds']))
        coverage=dict(completed_runs=len(main),expected_runs=expected_main,
                      complete=len(main)==expected_main)
        write_json(REPORT/'main_coverage.json',coverage)
        if not coverage['complete']:
            print('PARTIAL MAIN STUDY: do not rank unbalanced method averages.',coverage)
        table_text.append(('Main coverage (unbalanced partial averages must not be ranked)',json.dumps(coverage,indent=2)))
        # Average frames within sequence before aggregating sequences. Seeds
        # describe initialization sensitivity, not additional independent scenes.
        scene_seed=main.groupby(['method','scene','seed'],as_index=False).agg(
            unobserved_absrel_pct=('unobserved_absrel_pct','mean'),all_absrel_pct=('all_absrel_pct','mean'),
            observed_absrel_pct=('observed_absrel_pct','mean'),unobserved_rmse=('unobserved_rmse','mean'),
            adaptation_seconds=('adaptation_seconds','mean'),trainable_parameters=('trainable_parameters','first'),
            frames=('sample','nunique'),peak_memory_GB=('peak_memory_GB','max'))
        scene_seed.to_csv(REPORT/'main_scene_seed.csv',index=False)
        scene_mean=scene_seed.groupby(['method','scene'],as_index=False).agg(
            unobserved_absrel_pct=('unobserved_absrel_pct','mean'),seed_sd=('unobserved_absrel_pct','std'),
            all_absrel_pct=('all_absrel_pct','mean'),observed_absrel_pct=('observed_absrel_pct','mean'),
            adaptation_seconds=('adaptation_seconds','mean'),trainable_parameters=('trainable_parameters','first'),
            seeds=('seed','nunique'),frames=('frames','min'),peak_memory_GB=('peak_memory_GB','max'))
        scene_mean.to_csv(REPORT/'main_scene_means.csv',index=False)
        summary_main=scene_mean.groupby('method').agg(
            mean_scene_absrel_pct=('unobserved_absrel_pct','mean'),scene_sd=('unobserved_absrel_pct','std'),
            scenes=('scene','nunique'),min_seeds=('seeds','min'),min_frames_per_scene=('frames','min'),
            mean_adaptation_seconds=('adaptation_seconds','mean'),trainable_parameters=('trainable_parameters','first'),
            peak_memory_GB=('peak_memory_GB','max'))
        summary_main.to_csv(REPORT/'main_summary.csv')
        print('\nMAIN SUMMARY (check coverage columns):\n',summary_main.to_string())
        table_text.append(('Main test summary',summary_main.to_string()))
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        histories=ALL[ALL.stage=='test']
        for method in METHODS if 'METHODS' in globals() else sorted(main.method.unique()):
            a=histories[histories.method==method]
            if not len(a):continue
            # Equal frames were selected per scene; curves are descriptive means.
            curve=a.groupby(['scene','updates'],as_index=False)[['unobserved_absrel_pct','sparse_L1']].mean()
            curve=curve.groupby('updates')[['unobserved_absrel_pct','sparse_L1']].mean()
            axs[0].plot(curve.index,curve.unobserved_absrel_pct,label=method)
            axs[1].plot(curve.index,curve.sparse_L1,label=method)
        base=main[main.method=='frozen']
        if len(base):axs[0].axhline(base.unobserved_absrel_pct.mean(),ls=':',color='black',label='aligned base')
        for ax in axs:ax.set_xlabel('Successful updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axs[0].set_ylabel('Unobserved AbsRel (%)');axs[1].set_ylabel('Supplied-depth L1')
        fig.suptitle('Single-frame ScanNet pilot: completed runs only; see coverage tables')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(REPORT/f'main_convergence.{ext}',dpi=180)
        display(fig);plt.close(fig)
        # Predeclared descriptive target: 25% reduction from aligned-base error.
        targets=[]
        for sample in base['sample'].unique():
            threshold=.75*float(base[base['sample']==sample].unobserved_absrel_pct.iloc[0])
            for folder,a in histories[(histories['sample']==sample)&(histories.method!='frozen')].groupby('folder'):
                a=a.sort_values('updates');reached=a[a.unobserved_absrel_pct<=threshold]
                first=reached.iloc[0] if len(reached) else None
                targets.append(dict(sample=sample,method=a.method.iloc[0],seed=int(a.seed.iloc[0]),
                    target_absrel_pct=threshold,reached=first is not None,
                    first_recorded_update=float(first.updates) if first is not None else np.nan,
                    time_to_prediction_seconds=float(first.time_to_prediction_seconds) if first is not None else np.nan))
        pd.DataFrame(targets).to_csv(REPORT/'time_to_25pct_improvement.csv',index=False)
        # A common-scale qualitative comparison for the first test frame.
        first_sample=next((r['id'] for r in records_for('test') if r['id'] in set(main['sample'])),None)
        if first_sample:
            rec=record_lookup[first_sample];rgb,gt,valid=load_frame(rec)
            subset=main[(main['sample']==first_sample)&(main.seed==47)]
            picks=['frozen','lora4','learned','random','hybrid']
            predictions=[]
            for method in picks:
                a=subset[subset.method==method]
                if len(a):
                    result=torch.load(ROOT/a.iloc[0].folder/'result.pt',map_location='cpu',weights_only=False)
                    predictions.append((method,result['prediction'][0].numpy()))
            if predictions:
                truth=gt[0].numpy();vm=valid[0].numpy()
                vmax=float(np.percentile(truth[vm],98))
                errors=[np.where(vm,np.abs(p-truth),np.nan) for _,p in predictions]
                emax=max(float(np.nanpercentile(np.stack(errors),95)),1e-4)
                fig,axs=plt.subplots(2,len(predictions),figsize=(3*len(predictions),6),squeeze=False)
                for col,((method,p),e) in enumerate(zip(predictions,errors)):
                    axs[0,col].imshow(p,cmap='viridis',vmin=0,vmax=vmax);axs[0,col].set_title(method)
                    axs[1,col].imshow(e,cmap='magma',vmin=0,vmax=emax)
                for ax in axs.flat:ax.axis('off')
                fig.suptitle(first_sample+' | common depth and absolute-error color scales')
                fig.tight_layout();fig.savefig(REPORT/'qualitative.png',dpi=180);plt.close(fig)
    robustness=FINALS[FINALS.stage=='robustness']
    if len(robustness):
        robust_scene=robustness.groupby(['method','scene','points','noise'],as_index=False).unobserved_absrel_pct.mean()
        robust_summary=robust_scene.groupby(['method','points','noise']).agg(
            mean_absrel_pct=('unobserved_absrel_pct','mean'),scenes=('scene','nunique'))
        robust_summary.to_csv(REPORT/'robustness_summary.csv')
        table_text.append(('Robustness (30-update budget; partial coverage explicitly listed)',robust_summary.to_string()))
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for method,a in robust_scene.groupby('method'):
            clean=a[a.noise==0].groupby('points').unobserved_absrel_pct.mean()
            noisy=a[a.points==100].groupby('noise').unobserved_absrel_pct.mean()
            if len(clean):axs[0].plot(clean.index,clean.values,marker='o',label=method)
            if len(noisy):axs[1].plot(100*noisy.index,noisy.values,marker='o',label=method)
        axs[0].set_xlabel('Supplied depth points');axs[0].set_xscale('log')
        axs[1].set_xlabel('Corrupted conditioning points (%)')
        for ax in axs:ax.set_ylabel('Unobserved AbsRel (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(REPORT/f'robustness.{ext}',dpi=180)
        display(fig);plt.close(fig)

if (ROOT/'attribute_validation.csv').exists():
    attributes=pd.read_csv(ROOT/'attribute_validation.csv')
    ab=attributes.groupby('attribute_basis').unobserved_absrel_pct.mean()
    table_text.append(('Attribute bases: validation diagnostic only',ab.to_string()))
    fig,ax=plt.subplots(figsize=(8,4));ab.plot.bar(ax=ax)
    ax.set_ylabel('Validation unobserved AbsRel (%)');ax.set_xlabel('Basis source (equal dimensions)')
    fig.tight_layout();fig.savefig(REPORT/'attribute_ablation.png',dpi=180);plt.close(fig)
if (ROOT/'attribute_basis_overlap.csv').exists():
    ov=pd.read_csv(ROOT/'attribute_basis_overlap.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,field in zip(axs,['left_overlap','right_overlap']):
        matrix=ov.pivot(index='attribute_a',columns='attribute_b',values=field)
        im=ax.imshow(matrix.values,vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(len(matrix)),matrix.columns,rotation=45,ha='right')
        ax.set_yticks(range(len(matrix)),matrix.index);ax.set_title(field)
    fig.colorbar(im,ax=axs.tolist(),label='Normalized projection overlap')
    fig.savefig(REPORT/'attribute_overlap.png',dpi=180,bbox_inches='tight');plt.close(fig)

status=json.loads((ROOT/'stage_status.json').read_text()) if (ROOT/'stage_status.json').exists() else {}
description='''# DA4022 final evaluation: measured results

## Objective
Investigate whether fixed bases derived from controlled 3D variations can guide sparse-depth test-time adaptation, and whether a small residual LoRA restores flexibility.

## Data and separation
This is a small custom ScanNet single-frame adaptation study using the official CAPA sample tensors. Donor sequences: scene0707 and scene0766. Validation: scene0771. Test: scene0777 and scene0806. Two frames per sequence are used. No sequence crosses splits. The original sample conditioning is replaced by explicitly seeded random point measurements from valid dense GT. This is not the original CAPA sequence-level protocol or a standard zero-shot benchmark result.

## Controlled synthetic attributes
A simple analytic room/box/sphere renderer generates geometry, camera, texture, and illumination interventions. Within each adapter task, other scene parameters remain fixed and only the named attribute varies; one adapter is shared across the three variants. Two nuisance base scenes are used per attribute. This is low-diversity procedural data, not a reproduction of MegaSynth. Synthetic depth is known from the renderer. Test RGB/GT do not enter basis extraction.

## Methods
Frozen VGGT + sparse-depth alignment; ordinary image-encoder QKV LoRA rank 4; rank-1 LoRA as a near-budget comparator for the hybrid; random fixed bases; pooled synthetic attribute bases; real-donor bases; and synthetic bases + rank-1 residual LoRA. Main learned bases use a rank-d robust shared update mean, approximated by five p=1 IRLS iterations with low-rank QR/SVD factors. Orthonormal singular bases are retained for coefficient optimization. An additional learned_energy method pools Frobenius-normalized uncentered update energy as an extraction ablation. This is a small independent implementation inspired by the shared-update objective, not a full reproduction of the original paper's data and training pipeline. Four-dimensional left/right bases and a trainable 4x4 matrix per layer are used. Combined and individual attribute bases have equal dimensions, but combined bases use more donor tasks. LoRA and residual scaling are 2. All weight decay is zero. The coefficient and residual learning rates may differ; the exact choices are recorded.

## Selection, masks, and timing
Method learning rates are selected using final unobserved-region error on validation frames at a common 30-update budget, then frozen. Main evaluation uses up to 100 updates. GT is used for metrics and generating the declared sparse measurement simulation; no other dense test supervision enters gradients or alignment. Noise multiplies a seeded subset of observed values by uniform[0.2,2.0] factors. All methods share conditioning and corruption for an image/condition. Invalid predictions cause failed runs rather than exclusion. Nonpositive predictions remain in error metrics and their fractions are recorded. No test-best checkpoint is used for final scores.

Timings include synchronized forward/alignment/backward/optimizer work and exclude metrics, file saving, preprocessing and model loading. These are instrumented pilot timings with checkpointing and warm-cache effects, not deployment latency. Offline donor costs are available in the raw measurements. The reported time-to-target is the first recorded checkpoint reaching a predeclared 25% improvement over the aligned baseline; it is not time to a global optimum.

## Interpretation and limits
Tables are generated only from complete runs. Check scene/frame/seed coverage before comparing aggregates. Frames/seeds are not independent scenes. Only two real donor scenes, one validation scene, two test scenes and two synthetic base scenes are used: no broad generalization or statistical significance claim is justified. Seeds vary LoRA initialization and random bases; pure fixed-basis coefficient runs have no random initialization. CUDA reproducibility is best-effort, and warnings/versions are recorded. Attribute overlap is descriptive and does not establish semantic disentanglement. The quadratic experiment deliberately supplies an optimum-containing subspace and demonstrates a mechanism rather than a VGGT guarantee. Historical mid-evaluation numbers are not mixed into these tables. Any accuracy, convergence or robustness benefit must be supported by the generated results, not assumed.
'''
description+='\n## Stage completion\n\n```json\n'+json.dumps(status,indent=2)+'\n```\n'
description+='\nIncomplete/failed runs: '+str(len(failed))+' (see incomplete_runs.json).\n'
for title,text in table_text:
    description+='\n## '+title+'\n\n```text\n'+text+'\n```\n'
description+='''
## Presentation sequence
1. Proposal: supervision from CAPA, reusable directions from attribute mining.
2. Controlled known-optimum experiment and its misalignment failure.
3. Actual synthetic interventions, donor/validation/test separation, adapter equations.
4. Validation choices and main test accuracy/parameter/convergence tables.
5. Attribute ablation and learned-versus-random findings, including negative outcomes.
6. Residual-adapter, sparsity and outlier results.
7. Limitations, offline cost, failure cases and conclusions limited to the measured evidence.

## Sources
- CAPA code and license: https://github.com/nv-dvl/capa
- VGGT: https://github.com/facebookresearch/vggt
- Attribute-subspace paper: https://arxiv.org/abs/2604.10095
- Original course proposal and historical findings are maintained separately in the local workspace.
'''
(REPORT/'FINAL_REPORT.md').write_text(description,encoding='utf-8')
import html
images=''.join(f'<h2>{html.escape(p.stem)}</h2><img src="{p.name}" style="max-width:100%">' for p in sorted(REPORT.glob('*.png')))
(REPORT/'FINAL_REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>DA4022 results</title>'
    '<style>body{font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px}pre{white-space:pre-wrap}</style>'
    '<pre>'+html.escape(description)+'</pre>'+images,encoding='utf-8')
journal('report_generated',stage_status=status,completed_measurements=len(ALL),incomplete_runs=len(failed))
print('Reports:',REPORT/'FINAL_REPORT.md',REPORT/'FINAL_REPORT.html')
print('Important: inspect generated tables. No method is assumed to win.')
backup()
