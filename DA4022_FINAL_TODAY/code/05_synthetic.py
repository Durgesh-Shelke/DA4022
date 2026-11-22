# A small controlled renderer supplies actual scene/camera/material/light
# interventions. It is intentionally simple, not MegaSynth or photorealistic.
SYNTH = ROOT / 'synthetic'
SYNTH.mkdir(exist_ok=True)

def render_controlled(seed, attribute, level, height=240, width=320):
    assert attribute in CFG['synthetic_attributes']
    rng = np.random.default_rng(seed)
    # Base scene parameters are identical across attributes for a given seed.
    colors = rng.uniform(.25,.85,size=(7,3)).astype(np.float32)
    center = np.array([-.7 + rng.uniform(-.2,.2), -.65, 3.4],dtype=np.float32)
    radius = .65
    box_center = np.array([.85,-.8,4.1],dtype=np.float32)
    box_half = np.array([.55,.65,.55],dtype=np.float32)
    camera = np.array([0.,0.,0.],dtype=np.float32)
    target = np.array([0.,-.1,4.],dtype=np.float32)
    light = np.array([-1.,2.,-2.],dtype=np.float32)
    texture_frequency, texture_contrast = 2.0, .22
    if attribute=='geometry':
        radius *= 1 + .25*level
        center[0] += .25*level
        box_half[1] *= 1 + .25*level
    elif attribute=='camera':
        camera[0] += .35*level
        camera[1] += .12*level
    elif attribute=='texture':
        texture_frequency = 2.0 ** (level+1)
    elif attribute=='lighting':
        light[0] += 2.2*level
        light[2] += .5*level
    forward = target-camera; forward /= np.linalg.norm(forward)
    right = np.cross(forward,np.array([0.,1.,0.])); right /= np.linalg.norm(right)
    # For forward +Z, cross(forward,up) is -X; use a consistent right-handed camera.
    right = -right
    up = np.cross(forward,right); up /= np.linalg.norm(up)
    yy,xx = np.mgrid[0:height,0:width]
    u = (2*(xx+.5)/width-1)*np.tan(np.deg2rad(60)/2)
    v = (1-2*(yy+.5)/height)*np.tan(np.deg2rad(60)/2)*(height/width)
    rays = forward + u[...,None]*right + v[...,None]*up
    depth = np.full((height,width),np.inf)
    normal = np.zeros((height,width,3))
    material = np.zeros((height,width),dtype=np.int32)
    def install(t,n,index):
        take = (t>1e-3)&(t<depth)
        depth[take]=t[take]
        normal[take]=np.broadcast_to(n,normal.shape)[take]
        material[take]=index
    # Back wall, floor, ceiling, and side walls.
    planes = [(2,6.,[0,0,-1],0),(1,-1.5,[0,1,0],1),
              (1,2.,[0,-1,0],2),(0,-3.,[1,0,0],3),(0,3.,[-1,0,0],4)]
    for axis,value,n,index in planes:
        denominator = rays[...,axis]
        denominator = np.where(np.abs(denominator)<1e-8,1e-8,denominator)
        install((value-camera[axis])/denominator,np.array(n),index)
    # Sphere intersection, using unnormalised rays (t is camera-Z depth).
    oc = camera-center
    aa = (rays*rays).sum(-1)
    bb = 2*(rays*oc).sum(-1)
    cc = (oc*oc).sum()-radius**2
    discriminant = bb**2-4*aa*cc
    t = (-bb-np.sqrt(np.maximum(discriminant,0)))/(2*aa)
    t = np.where(discriminant>0,t,np.inf)
    finite_t = np.where(np.isfinite(t),t,0)
    normals = (camera + finite_t[...,None]*rays-center)/radius
    install(t,normals,5)
    # Axis-aligned box using slab intersections.
    safe_rays = np.where(np.abs(rays)<1e-8,1e-8,rays)
    t0 = (box_center-box_half-camera)/safe_rays
    t1 = (box_center+box_half-camera)/safe_rays
    enter = np.minimum(t0,t1).max(-1)
    leave = np.maximum(t0,t1).min(-1)
    box_t = np.where((leave>=enter)&(enter>0),enter,np.inf)
    p = camera + np.where(np.isfinite(box_t),box_t,0)[...,None]*rays
    local = (p-box_center)/box_half
    axis = np.abs(local).argmax(-1)
    box_n = np.zeros_like(rays)
    for j in range(3):
        box_n[...,j] = np.where(axis==j,np.sign(local[...,j]),0)
    install(box_t,box_n,6)
    points = camera+depth[...,None]*rays
    pattern = np.sin(texture_frequency*points[...,0]*np.pi)*np.sin(texture_frequency*points[...,2]*np.pi)
    albedo = colors[material]*(1+texture_contrast*pattern[...,None])
    light /= np.linalg.norm(light)
    lambert = np.maximum((normal*light).sum(-1),0)
    rgb = np.clip(albedo*(.3+.7*lambert[...,None]),0,1)
    valid = np.isfinite(depth)&(depth>0)&(depth<50)
    return rgb.transpose(2,0,1).astype('float32'),np.where(valid,depth,0).astype('float32'),valid

SYNTH_TASKS = []
for attribute in CFG['synthetic_attributes']:
    for seed in CFG['synthetic_base_seeds']:
        task = []
        for level in CFG['synthetic_levels']:
            sample_id = f'synth_{attribute}_{seed}_{level:+d}'
            file = PREPARED/(sample_id+'.npz')
            rgb,gt,valid = render_controlled(seed,attribute,level)
            np.savez_compressed(file,rgb=rgb,gt=gt,valid=valid)
            task.append(dict(id=sample_id,scene=f'synthetic_base_{seed}',role='synthetic_donor',
                attribute=attribute,level=level,file=str(file.relative_to(ROOT)),sha256=sha256(file)))
        SYNTH_TASKS.append(dict(id=f'{attribute}_{seed}',attribute=attribute,frames=task))
write_json(ROOT/'synthetic_manifest.json',dict(tasks=SYNTH_TASKS,
    description='Analytic room/box/sphere ray casting; Lambertian shading; controlled target intervention; nuisance scene seed fixed within each task.'))
fig,axs=plt.subplots(4,3,figsize=(10,10))
for row,attribute in enumerate(CFG['synthetic_attributes']):
    task=next(t for t in SYNTH_TASKS if t['attribute']==attribute)
    for ax,rec in zip(axs[row],task['frames']):
        rgb,_,_=load_frame(rec)
        ax.imshow(rgb[0].permute(1,2,0));ax.set_title(attribute+f' level {rec["level"]}');ax.axis('off')
fig.tight_layout();fig.savefig(SYNTH/'controlled_variations.png',dpi=160)
display(fig);plt.close(fig)
journal('synthetic_data_ready',tasks=len(SYNTH_TASKS),attributes=CFG['synthetic_attributes'])
