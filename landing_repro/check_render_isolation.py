from pathlib import Path
import json
import numpy as np
from landing_repro.environment import MovingPlatformLanding
from landing_repro.motion_curriculum import motion_config
from landing_repro.smooth import smooth_config
from landing_repro.center_marker import update_center_marker
from landing_repro.scene_views import inspection_views

cfg=motion_config(smooth_config,2,True,True,True)
a,b=MovingPlatformLanding(cfg),MovingPlatformLanding(cfg)
try:
    oa,_=a.reset(seed=82001); ob,_=b.reset(seed=82001)
    for i in range(30):
        action=np.array([.05,-.03,-.35],np.float32)
        oa,ra,ta,ua,ia=a.step(action)
        ob,rb,tb,ub,ib=b.step(action)
        for _ in range(3): update_center_marker(a)
        if i%5==0: inspection_views(a)
        assert a.center_estimate==b.center_estimate, i
        assert np.array_equal(oa['pixels'],ob['pixels']),i
        assert np.allclose(a.state,b.state,rtol=0,atol=1e-12),i
        assert (ra,ta,ua)==(rb,tb,ub),i
    result={'status':'passed','steps':30,'display_refreshes':90,'inspection_frames':6,
            'check':'Repeated rendering leaves estimator, observations, physics and reward unchanged'}
    Path('runs/height_curriculum_checks').mkdir(parents=True,exist_ok=True)
    Path('runs/height_curriculum_checks/render_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
finally:
    a.close();b.close()
