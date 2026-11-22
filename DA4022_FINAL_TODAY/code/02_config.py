# Change settings BEFORE the first experiment. Changed scientific settings
# require a new ROOT directory; this prevents silently mixing experiments.
CFG = {
    'protocol_version': 'today-v1',
    'input_width': 518,
    'frames_per_scene': 2,
    'donor_steps': 40,
    'validation_steps': 30,
    'test_steps': 100,
    'robustness_steps': 30,
    'rank': 4,
    'basis_dim': 4,
    'basis_extraction': 'robust_shared_mean_irls',
    'irls_iterations': 5,
    'residual_rank': 1,
    'donor_lr': 0.001,
    'weight_decay': 0.0,
    'align_points': 1000,
    'main_seeds': [47, 48],
    'conditioning_seed': 20261122,
    'donor_variants': ['identity'],
    'synthetic_attributes': ['geometry', 'camera', 'texture', 'lighting'],
    'synthetic_base_seeds': [310, 311],
    'synthetic_levels': [-1, 0, 1],
    'max_gpu_minutes': 150,
    'run_robustness': True,
    'run_appearance_ablation': True,
    'main_conditions': {'points': 100, 'noise': 0.0},
    'lr_candidates': {
        'lora4': [0.0003, 0.001],
        'lora1': [0.0003, 0.001],
        'learned': [0.01, 0.1],
        'learned_energy': [0.01, 0.1],
        'random': [0.01, 0.1],
        'hybrid': [0.01, 0.1],
        'shared_real': [0.01, 0.1],
    },
    'hybrid_residual_lr': 0.001,
    'split': {
        'donor': ['scene0707', 'scene0766'],
        'validation': ['scene0771'],
        'test': ['scene0777', 'scene0806'],
    },
}
# This is a small, custom single-frame adaptation protocol. Frames from a
# sequence stay in one split, and scene means (not frame means) are reported.
# It is NOT the original CAPA multi-frame evaluation or a large benchmark.
scientific_cfg = {k: v for k, v in CFG.items() if k != 'max_gpu_minutes'}
scientific_cfg['build_hash'] = globals().get('BUILD_HASH','manual-source-v1')
CONFIG_HASH = hashlib.sha256(json.dumps(scientific_cfg, sort_keys=True).encode()).hexdigest()[:16]
config_path = ROOT / 'config.json'
if config_path.exists():
    previous = json.loads(config_path.read_text())
    assert previous['config_hash'] == CONFIG_HASH, (
        'Scientific settings changed. Use a different ROOT or restore the original CFG.')
write_json(config_path, {'config_hash': CONFIG_HASH, 'config': CFG, 'capa_commit': CAPA_COMMIT})
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)
SESSION_START = time.monotonic()
class BudgetStop(RuntimeError):
    pass
def check_budget():
    if (time.monotonic() - SESSION_START) / 60 > CFG['max_gpu_minutes']:
        raise BudgetStop('Session compute budget reached. Run the reporting and backup cells now. Resume later with the saved folder.')
print('Config:', CONFIG_HASH)
print('Donors / validation / test:', CFG['split'])
print('Main study:', len(CFG['main_seeds']), 'seeds; 2 test scenes; 2 frames per scene.')
