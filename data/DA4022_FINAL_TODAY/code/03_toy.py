# Controlled mechanism experiment. No GPU or pretrained model required.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOY = ROOT / 'controlled'
TOY.mkdir(exist_ok=True)
toy_curves = {m: [] for m in ['Full space', 'Aligned', 'Misaligned', 'Random']}
toy_rows = []
n, d, iterations, tolerance = 60, 6, 1500, 1e-6
for seed in range(10):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    H = (Q * np.geomspace(1, 1e4, n)) @ Q.T
    pair = np.linalg.qr(rng.normal(size=(n, 2*d)))[0]
    U, P = pair[:, :d], pair[:, d:]
    optimum = U @ rng.normal(size=d)
    optimum /= np.linalg.norm(optimum)
    bases = {'Full space': np.eye(n), 'Aligned': U,
             'Misaligned': np.cos(np.pi/6)*U + np.sin(np.pi/6)*P,
             'Random': np.linalg.qr(rng.normal(size=(n,d)))[0]}
    initial = .5 * optimum @ H @ optimum
    for method, V in bases.items():
        h, rhs = V.T @ H @ V, V.T @ H @ optimum
        eig = np.linalg.eigvalsh(h)
        eta = 2 / (eig[0] + eig[-1])
        z = np.zeros(V.shape[1])
        exact_error = V @ np.linalg.solve(h, rhs) - optimum
        attainable = .5 * exact_error @ H @ exact_error / initial
        curve = []
        for t in range(iterations+1):
            error = V @ z - optimum
            curve.append(max(0., float(.5 * error @ H @ error / initial)))
            z -= eta * (h @ z - rhs)
        toy_curves[method].append(curve)
        reached = np.flatnonzero(np.array(curve) <= tolerance)
        toy_rows.append(dict(seed=seed, method=method, condition_number=eig[-1]/eig[0],
                             final_gap=curve[-1], attainable_gap=max(0., float(attainable)),
                             steps_to_tolerance=int(reached[0]) if len(reached) else np.nan,
                             reached=bool(len(reached))))
toy_df = pd.DataFrame(toy_rows)
toy_df.to_csv(TOY/'results.csv', index=False)
toy_summary = toy_df.groupby('method').agg(success_rate=('reached','mean'),
    median_steps=('steps_to_tolerance','median'), median_final_gap=('final_gap','median'),
    median_condition=('condition_number','median'))
toy_summary.to_csv(TOY/'summary.csv')
fig, axs = plt.subplots(1,2,figsize=(12,4))
for method, curves in toy_curves.items():
    a = np.array(curves)
    for ax in axs:
        line, = ax.plot(np.maximum(np.median(a,axis=0),1e-16), label=method)
        ax.fill_between(np.arange(iterations+1), np.maximum(np.percentile(a,25,axis=0),1e-16),
            np.maximum(np.percentile(a,75,axis=0),1e-16), alpha=.12,color=line.get_color())
for ax in axs:
    ax.set_yscale('log'); ax.axhline(tolerance,ls=':',color='black')
    ax.set_xlabel('Updates'); ax.set_ylabel('Loss gap / initial gap'); ax.grid(alpha=.2); ax.legend()
axs[0].set_xlim(0,100); axs[0].set_title('Early convergence')
axs[1].set_title('Longer convergence and irreducible error')
fig.suptitle('Known-optimum quadratic: constructed subspaces, 10 seeds')
fig.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(TOY/f'convergence.{ext}',dpi=180)
display(fig); plt.close(fig)
print(toy_summary)
journal('controlled_experiment_complete', output='controlled/results.csv')
backup()
