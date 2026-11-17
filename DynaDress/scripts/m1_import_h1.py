"""Convert the audited task URDF using installed Isaac 6 importer."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.urdf')
    app.update()
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    cfg=URDFImporterConfig(urdf_path=str(ROOT/'assets/h1/generated/h1_task.urdf'),usd_path=str(ROOT/('assets/h1/generated/usd_decomposed' if '--decomposed' in sys.argv else 'assets/h1/generated/usd')),collision_type='Convex Decomposition' if '--decomposed' in sys.argv else 'Convex Hull',fix_base=True,merge_fixed_joints=True,allow_self_collision=True,joint_drive_type='force',joint_target_type='position',override_joint_stiffness=500.,override_joint_damping=40.)
    importer=URDFImporter();importer.config=cfg
    path=importer.import_urdf()
    (ROOT/'assets/h1/generated/usd_path.txt').write_text(path)
    print('M1-H1-IMPORT-PASS '+path,flush=True)
except BaseException:
    import traceback,os
    traceback.print_exc()
    app.close(exit_code=1)
    os._exit(1)
finally:
    app.close()
