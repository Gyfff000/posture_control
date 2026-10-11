"""PerchRL-inspired periodic cubic B-splines with bounded OU speed variation.

Environment trajectory only: samples/future waypoints never enter actor inputs.
The horizontal platform translates and yaws with its travel tangent, as a cart.
"""
import numpy as np


def spline(controls,u):
    u=np.asarray(u); j=np.floor(u).astype(int); t=u-j
    points=controls[(j[...,None]+np.array([-1,0,1,2]))%len(controls)]
    b=np.stack([(1-t)**3,3*t**3-6*t*t+4,-3*t**3+3*t*t+3*t+1,t**3],-1)/6
    d=np.stack([-3*(1-t)**2,9*t*t-12*t,-9*t*t+6*t+3,3*t*t],-1)/6
    dd=np.stack([6*(1-t),18*t-12,-18*t+6,6*t],-1)/6
    return tuple(np.sum(points*w[...,None],axis=-2) for w in (b,d,dd))


class PlatformMotion:
    def __init__(self,cfg,rng,origin,speed,yaw):
        if cfg.platform_motion not in ('constant','variable_speed','random_bspline'):
            raise ValueError('Unknown platform_motion: '+cfg.platform_motion)
        if not 0<=cfg.platform_speed_min<=cfg.platform_speed_max:
            raise ValueError('Invalid speed bounds')
        self.mode=cfg.platform_motion; self.origin=np.array(origin); self.yaw=yaw
        self.initial_velocity=np.array([speed*np.cos(yaw),speed*np.sin(yaw),0.])
        self.dt=1/cfg.physics_hz; self.max_curvature=0.
        if self.mode=='constant':
            return
        if cfg.platform_accel_max<=0 or cfg.platform_speed_noise<0:
            raise ValueError('Variable motion requires positive acceleration and nonnegative noise')
        n=int(np.ceil((cfg.episode_seconds+2)*cfg.physics_hz))+1
        self.times=np.arange(n)*self.dt
        v=np.empty(n); v[0]=np.clip(speed,cfg.platform_speed_min,cfg.platform_speed_max)
        eta=0.; a=0.; beta=self.dt/(2.+self.dt)
        for k in range(1,n):
            eta=(1-beta)*eta+cfg.platform_speed_noise*np.sqrt(beta)*rng.normal()
            target=np.clip(max(speed,.01)*np.exp(np.clip(eta,-3,3)),cfg.platform_speed_min,cfg.platform_speed_max)
            desired=np.clip((target-v[k-1])/.5,-.5*cfg.platform_accel_max,.5*cfg.platform_accel_max)
            a+=np.clip(desired-a,-.5*self.dt,.5*self.dt)
            v[k]=np.clip(v[k-1]+a*self.dt,cfg.platform_speed_min,cfg.platform_speed_max)
        distance=np.r_[0.,np.cumsum((v[1:]+v[:-1])*.5*self.dt)]
        if self.mode=='variable_speed':
            direction=np.array([np.cos(yaw),np.sin(yaw)])
            xy=distance[:,None]*direction; heading=np.full(n,yaw)
        else:
            if cfg.platform_turn_rate_degrees<=0 or cfg.platform_speed_max<=0:
                raise ValueError('Curved motion requires positive turn and speed limits')
            # Use peak speed, rather than nominal speed, so perturbations retain margins.
            vmax=cfg.platform_speed_max
            limit=.95*min(np.deg2rad(cfg.platform_turn_rate_degrees)/vmax,
                           .75*cfg.platform_accel_max/vmax**2)
            m=12; angles=np.arange(m)*2*np.pi/m
            radii=rng.uniform(.7,1.3,m)
            for _ in range(3):
                radii=.25*np.roll(radii,1)+.5*radii+.25*np.roll(radii,-1)
            mean=max(2.5,1.5/limit); deviations=(radii-radii.mean())*mean
            u=np.linspace(0,m,8193)
            def candidate(alpha):
                r=mean+alpha*deviations
                controls=np.c_[r*np.cos(angles),r*np.sin(angles)]
                xy,d,dd=spline(controls,u)
                curvature=np.abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum(np.linalg.norm(d,axis=1)**3,1e-12)
                return controls,xy,d,float(curvature.max())
            low,high=0.,1.
            if candidate(high)[3]>limit:
                for _ in range(25):
                    mid=(low+high)/2
                    if candidate(mid)[3]<=limit: low=mid
                    else: high=mid
                high=low
            controls,path,tangent,self.max_curvature=candidate(high)
            if self.max_curvature>limit*(1+1e-6):
                raise ValueError('Unable to satisfy sampled curvature bound')
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            parameter=np.interp(distance%arc[-1],arc,u)
            xy,d,_=spline(controls,parameter)
            first=spline(controls,np.array([0.]))
            rotation=yaw-np.arctan2(first[1][0,1],first[1][0,0])
            co,si=np.cos(rotation),np.sin(rotation); matrix=np.array([[co,-si],[si,co]])
            xy=(xy-first[0][0])@matrix.T
            heading=np.unwrap(np.arctan2(d[:,1],d[:,0]))+rotation
        self.positions=self.origin+np.c_[xy,np.zeros(n)]
        self.headings=heading
        self.velocities=np.c_[v*np.cos(heading),v*np.sin(heading),np.zeros(n)]
        self.initial_velocity=self.velocities[0].copy()
        self.yaw_rates=np.gradient(heading,self.dt)

    def sample(self,t):
        if self.mode=='constant':
            return self.origin+self.initial_velocity*t,self.initial_velocity.copy(),self.yaw,0.
        if t<0 or t>self.times[-1]+1e-8:
            raise ValueError('Trajectory requested outside generated duration')
        tick=int(np.clip(round(t/self.dt),0,len(self.times)-1))
        return self.positions[tick].copy(),self.velocities[tick].copy(),float(self.headings[tick]),float(self.yaw_rates[tick])
