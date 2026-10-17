"""Paper-task monitor. External views never enter observations or rewards."""
import json,os,time
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import pybullet as p
from PIL import Image,ImageTk

class Monitor:
    def __init__(self,out,evaluation=False):
        self.evaluation=evaluation
        self.out=Path(out);self.paused=False;self.stop=False;self.last_draw=0.;self.last_status=0.
        self.root=tk.Tk();self.root.title(('UAV LANDING FROZEN TEST' if evaluation else 'UAV LANDING TRAINING')+' | Ladosz 2024 paper-derived protocol')
        tk.Label(self.root,text=('FROZEN MODEL TEST | NO TRAINING / WEIGHT UPDATES' if evaluation else 'PAPER-DERIVED DrQ-v2 | FROM SCRATCH | NO EXPERT / PREDICTION / NEAR CONTROL'),font=('Arial',15,'bold')).pack(pady=6)
        tk.Label(self.root,text='10 +/- 3 m above deck | 0-1 m/s constant platform | unresolved paper details are documented').pack()
        row=tk.Frame(self.root);row.pack();self.images=[]
        for title in ('Actual drone: external view','Moving platform: external view','Raw onboard camera: actor input'):
            col=tk.Frame(row);col.pack(side='left');tk.Label(col,text=title).pack();label=tk.Label(col);label.pack();self.images.append(label)
        self.stats=tk.Label(self.root,text='Initializing...',font=('Courier',12),justify='left');self.stats.pack(pady=8)
        row=tk.Frame(self.root);row.pack();self.button=tk.Button(row,text='Pause',command=self.toggle);self.button.pack(side='left',padx=10)
        tk.Button(row,text='Stop test' if evaluation else 'Stop and save',command=lambda:setattr(self,'stop',True)).pack(side='left',padx=10)
        self.log=scrolledtext.ScrolledText(self.root,width=110,height=6);self.log.pack(pady=8)
        tk.Label(self.root,text=str(self.out)).pack();self.root.protocol('WM_DELETE_WINDOW',lambda:setattr(self,'stop',True))
    def toggle(self):
        self.paused=not self.paused;self.button.configure(text='Continue' if self.paused else 'Pause')
    def pump(self):
        request=self.out/'show_window.request'
        if request.exists():
            request.unlink();self.root.deiconify();self.root.lift();self.root.attributes('-topmost',True);self.root.after(300,lambda:self.root.attributes('-topmost',False))
        try:self.root.update_idletasks();self.root.update()
        except tk.TclError:self.stop=True
    def wait(self):
        self.pump()
        while self.paused and not self.stop:self.pump();time.sleep(.05)
    def message(self,line):
        self.log.insert('end',line+'\n');self.log.see('end')
    def update(self,env,status,force=False):
        now=time.monotonic();status=dict(status,pid=os.getpid(),paused=self.paused)
        if force or now-self.last_status>=1.:
            tmp=self.out/'live_status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(self.out/'live_status.json');self.last_status=now
        if force or now-self.last_draw>=.5:
            frames=[]
            for target,distance,pitch in ((env.state[:3],.38,-25),(env.platform_position+[0,0,.04],1.65,-55)):
                view=p.computeViewMatrixFromYawPitchRoll(target,distance,45,pitch,0,2)
                projection=p.computeProjectionMatrixFOV(55,1.25,.005,150)
                image=p.getCameraImage(350,280,view,projection,renderer=p.ER_TINY_RENDERER,physicsClientId=env.client)[2]
                frames.append(Image.fromarray(np.asarray(image,np.uint8).reshape(280,350,4)[...,:3]))
            frames.append(Image.fromarray(env.frames[-1]).convert('RGB').resize((280,280),Image.Resampling.NEAREST))
            for label,frame in zip(self.images,frames):
                shown=ImageTk.PhotoImage(frame);label.configure(image=shown);label.image=shown
            self.stats.configure(text=f"{status['state']} | Seed {status['seed']} | Step {status['step']:,} | Updates {status['updates']:,}\n" +
                (f"Test episodes {status['episodes']} | Test successes {status['successes']} | Optimization OFF\n" if self.evaluation else f"Episodes {status['episodes']} / 4000 | Training contacts {status['successes']} | Replay {status['replay_size']:,}\n") +
                f"Height above deck {env.state[2]-env.cfg.platform_top:.2f} m | Sim {env.elapsed:.2f}s | Warm-up {status['warmup']}\n"
                f"Critic loss {status.get('critic_loss','pending')} | {status.get('last_event','running')}\n"
                'Camera estimator OFF | Near control OFF | Demonstrations OFF | Smooth shaping OFF')
            env.update_debug_camera();self.last_draw=now
        self.pump()
    def close(self):
        try:self.root.destroy()
        except tk.TclError:pass