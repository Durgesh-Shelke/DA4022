import requests, gc

SAMPLE_BASE = 'https://share.phys.ethz.ch/~pf/bingkedata/capa/sample_data/scannet_sift_noise_10pct/'
PREPARED = ROOT / 'data'
PREPARED.mkdir(exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def download_sample(scene):
    # Reuse an attached official sample when available.
    attached = sorted(Path('/kaggle/input').rglob(scene + '.pt'))
    if attached:
        return attached[0]
    target = DATA_CACHE / (scene + '.pt')
    if target.exists():
        return target
    part = target.with_suffix('.pt.part')
    url = SAMPLE_BASE + scene + '.pt'
    for attempt in range(2):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(25,90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                total = response.headers.get('Content-Length','unknown')
                print('Downloading', scene, 'response bytes:', total, flush=True)
                with part.open('ab' if append else 'wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk)
            part.replace(target)
            return target
        except Exception as exc:
            journal('download_error', scene=scene, error=str(exc), attempt=attempt)
            if attempt == 1:
                raise RuntimeError(f'Could not download {scene}. Attach the official .pt file as a Kaggle input and rerun this cell.') from exc

manifest_path = ROOT / 'data_manifest.json'
if manifest_path.exists():
    MANIFEST = json.loads(manifest_path.read_text())
    assert MANIFEST['config_hash'] == CONFIG_HASH
    for record in MANIFEST['frames']:
        file = ROOT / record['file']
        assert file.exists() and sha256(file) == record['sha256'], f'Missing/changed prepared frame: {file}'
    print('Reusing verified prepared frames.')
else:
    records, sources = [], []
    for role, scenes in CFG['split'].items():
        for scene in scenes:
            source = download_sample(scene)
            # Official/public CAPA tensors or the user's attached copies only.
            pack = torch.load(source, map_location='cpu', weights_only=False)
            assert 'depth_gt_nvhw' in pack, f'{scene} has no dense GT; cannot use for this study.'
            rgb = pack['rgb_nv3hw'].float()
            truth = pack['depth_gt_nvhw'].float()
            H,W = rgb.shape[-2:]
            rgb = rgb.reshape(-1,3,H,W)
            truth = truth.reshape(-1,H,W)
            assert len(rgb) == len(truth)
            indices = np.unique(np.linspace(0,len(rgb)-1,CFG['frames_per_scene'],dtype=int))
            source_hash = sha256(source)
            sources.append(dict(scene=scene, role=role, url=SAMPLE_BASE+scene+'.pt',
                                sha256=source_hash, total_frames=len(rgb), selected_frames=indices.tolist()))
            for idx in indices:
                image = rgb[int(idx)].numpy()
                if np.nanmax(image) > 1.5:
                    image = image / 255.
                assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1.01
                gt = truth[int(idx)].numpy()
                valid = np.isfinite(gt) & (gt > 0)
                assert valid.sum() >= 1000, 'Insufficient valid depth.'
                # Preserve validity explicitly; no interpolation of GT or masks.
                gt = np.where(valid,gt,0).astype('float32')
                sample_id = f'{scene}_frame{int(idx):04d}'
                file = PREPARED / (sample_id+'.npz')
                np.savez_compressed(file, rgb=np.clip(image,0,1).astype('float32'), gt=gt, valid=valid)
                records.append(dict(id=sample_id,scene=scene,role=role,
                                    frame_index=int(idx),file=str(file.relative_to(ROOT)),sha256=sha256(file)))
            del pack,rgb,truth
            gc.collect()
            print('Prepared', role, scene, 'frames', indices.tolist(), flush=True)
    assert len({x['scene'] for x in records}) == sum(map(len,CFG['split'].values()))
    MANIFEST = dict(config_hash=CONFIG_HASH, protocol='single-frame adaptation; regenerated synthetic sparse depth',
                    frames=records,sources=sources)
    write_json(manifest_path, MANIFEST)

def records_for(role):
    return [r for r in MANIFEST['frames'] if r['role']==role]

def load_frame(record, variant='identity'):
    with np.load(ROOT / record['file']) as a:
        rgb,gt,valid = a['rgb'].copy(),a['gt'].copy(),a['valid'].copy()
    if variant=='dark':
        rgb = rgb * .6
    elif variant=='bright':
        rgb = np.clip(rgb * 1.4,0,1)
    elif variant!='identity':
        raise ValueError(variant)
    return torch.from_numpy(rgb[None]), torch.from_numpy(gt[None]), torch.from_numpy(valid[None])

def make_condition(record, points=100, noise=0., variant='identity'):
    rgb,gt,valid = load_frame(record,variant)
    # Same locations/noise across methods and training seeds. For different
    # point budgets the locations are nested by prefix of one permutation.
    key = f'{CFG["conditioning_seed"]}/{record["id"]}'
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(valid.numpy().reshape(-1))
    indices = rng.permutation(eligible)[:points]
    assert len(indices) == points
    mask = torch.zeros_like(valid).reshape(-1)
    mask[torch.from_numpy(indices.copy())] = True
    mask = mask.reshape_as(valid)
    condition = torch.where(mask,gt,torch.zeros_like(gt))
    noise_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    k = int(round(points*noise))
    selected = noise_rng.permutation(points)[:k]
    factors = noise_rng.uniform(.2,2.,size=points).astype('float32')
    corrupted = indices[selected]
    if k:
        flat = condition.reshape(-1)
        flat[torch.from_numpy(corrupted.copy())] *= torch.from_numpy(factors[selected].copy())
    return rgb,gt,valid,condition,mask,dict(points=points,noise_fraction=noise,
        actual_corrupted_points=k,seed=seed,noise_model='multiplicative replacement factor uniform[0.2,2.0]')

fig,axes = plt.subplots(len(CFG['split']),4,figsize=(12,8),squeeze=False)
for row,role in enumerate(CFG['split']):
    for ax in axes[row]: ax.axis('off')
    for ax,rec in zip(axes[row],records_for(role)):
        image,_,_ = load_frame(rec)
        ax.imshow(image[0].permute(1,2,0)); ax.set_title(role+'\n'+rec['id'],fontsize=8)
fig.suptitle('Fixed splits: no sequence crosses donor / validation / test')
fig.tight_layout(); fig.savefig(ROOT/'split_contact_sheet.png',dpi=150)
display(fig); plt.close(fig)
print(pd.DataFrame(MANIFEST['frames'])[['id','scene','role']].to_string(index=False))
journal('data_ready',frames=len(MANIFEST['frames']),scenes=len(MANIFEST['sources']))
backup()
