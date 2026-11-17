"""Verify real Isaac startup, GPU tensor computation and FEM API presence."""
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
try:
    import importlib.metadata as md
    import json
    from pathlib import Path
    import torch
    from isaacsim.core.experimental.prims import DeformablePrim
    from omni.physx.scripts import deformableUtils
    result = {
        'isaac_version_file': Path('/isaac-sim/VERSION').read_text().strip(),
        'gpu': torch.cuda.get_device_name(0),
        'cuda_available': torch.cuda.is_available(),
        'gpu_tensor_sum': torch.ones(16, device='cuda').sum().item(),
        'torch_version': torch.__version__,
        'surface_deformable_helper': callable(deformableUtils.set_physics_surface_deformable_body),
        'deformable_class': str(DeformablePrim),
        'dependencies': {k: md.version(k) for k in ['termcolor', 'scipy', 'matplotlib', 'gymnasium', 'h5py']},
    }
    print('NATIVE-ISAAC-PROBE ' + json.dumps(result), flush=True)
    Path('/output/native_probe.json').write_text(json.dumps(result, indent=2))
except BaseException:
    import traceback
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()
