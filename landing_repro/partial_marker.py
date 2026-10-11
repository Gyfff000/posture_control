"""Track a previously identified planar marker through a clipped RGB view.

This is a short, fallible visual update, not a detector or a truth-state fallback.
Forward/backward optical flow and a nondegenerate RANSAC homography are required.
"""
import cv2
import numpy as np
import pybullet as p


class PartialMarkerTracker:
    def __init__(self):
        self.gray=None; self.points=None; self.center=None; self.quad=None

    def seed(self,rgb,measurement):
        if 'quad' not in measurement or 'center_pixel' not in measurement:
            self.gray=None
            return
        self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.center=np.asarray(measurement['center_pixel'],np.float32).reshape(1,1,2)
        self.quad=np.asarray(measurement['quad'],np.float32).reshape(-1,1,2)
        self.points=self.features(self.gray)

    def features(self,gray):
        mask=np.zeros_like(gray)
        # Coordinates are bounded before rasterization, including an offscreen quad.
        quad=np.clip(self.quad[:,0],-10000,10000).astype(np.int32)
        cv2.fillConvexPoly(mask,quad,255)
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        return cv2.goodFeaturesToTrack(gray,60,.015,3,mask=mask,blockSize=3)

    def update(self,rgb,own_position,own_quaternion,plane_z,config):
        if self.gray is None or self.points is None or len(self.points)<6:
            self.gray=None
            return None
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        new,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,
            winSize=(15,15),maxLevel=2)
        if new is None:
            self.gray=None
            return None
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,new,None,
            winSize=(15,15),maxLevel=2)
        h,w=gray.shape
        if back is None:
            self.gray=None
            return None
        good=(ok[:,0]>0)&(back_ok[:,0]>0)&(np.linalg.norm(back[:,0]-self.points[:,0],axis=1)<.7)
        good&=(new[:,0,0]>1)&(new[:,0,0]<w-2)&(new[:,0,1]>1)&(new[:,0,1]<h-2)
        old=self.points[good]; new=new[good]
        if len(new)<6 or cv2.contourArea(cv2.convexHull(old))<30:
            self.gray=None
            return None
        homography,inliers=cv2.findHomography(old,new,cv2.RANSAC,1.)
        if homography is None or inliers is None or inliers.mean()<.75:
            self.gray=None
            return None
        center=cv2.perspectiveTransform(self.center,homography)
        quad=cv2.perspectiveTransform(self.quad,homography)
        # Prevent unstable extrapolation from a narrow remaining sliver.
        area_ratio=abs(cv2.contourArea(quad))/max(abs(cv2.contourArea(self.quad)),1.)
        if (not np.isfinite(center).all() or not np.isfinite(quad).all()
                or not .5<area_ratio<2. or not -.5*w<center[0,0,0]<1.5*w
                or not -.5*h<center[0,0,1]<1.5*h):
            self.gray=None
            return None
        rotation=np.array(p.getMatrixFromQuaternion(own_quaternion)).reshape(3,3)
        forward=rotation@np.array([0.,0.,-1.]); up=rotation@np.array([1.,0.,0.])
        right=np.cross(forward,up)
        eye=np.asarray(own_position)+rotation@np.array([config.camera_offset_x,0.,config.camera_offset_z])
        f=h/(2*np.tan(np.deg2rad(config.camera_fov)/2))
        ray=np.column_stack([right,-up,forward])@np.array([
            (center[0,0,0]-w/2)/f,(center[0,0,1]-h/2)/f,1.])
        distance=(plane_z-eye[2])/ray[2] if abs(ray[2])>1e-6 else -1.
        if distance<=.04 or distance>3.:
            self.gray=None
            return None
        world=eye+distance*ray
        self.gray=gray; self.center=center; self.quad=quad
        self.points=self.features(gray)
        return dict(position=world,inlier_fraction=float(inliers.mean()),features=int(inliers.sum()))
