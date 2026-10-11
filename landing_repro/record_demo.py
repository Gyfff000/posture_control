"""Record actual frozen-policy PyBullet flights; include every requested seed."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
import torch

from .agent import DrQV2
from .center_marker import overlay,update_center_marker
from .config import EnvConfig
from .environment import MovingPlatformLanding,LIMITS
from .smooth import FlightLog

FONT='/mnt/c/Windows/Fonts/msyh.ttc'
BOLD='/mnt/c/Windows/Fonts/msyhbd.ttc'
FONTS={n:ImageFont.truetype(FONT,n) for n in (16,18,20,22,26)}
TITLE=ImageFont.truetype(BOLD,30)
RESULT=ImageFont.truetype(BOLD,40)
PAPER='Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning'
SOURCE={'full_marker':'完整标识检测','partial_flow':'局部光流跟踪','kalman_prediction':'卡尔曼预测','terminal_coast':'末段短时外推'}
EVENT={'success':'降落成功','near_estimate_lost':'中心估计失效','terminal_timeout':'末段超时','timeout':'飞行超时','lost_view':'目标丢失','platform_edge':'平台边缘接触','ground':'触地失败','escaped':'偏离平台','running':'测试进行中'}


def background():
    return Image.new('RGB',(1280,720),(245,247,250))


def label(draw,position,text,size=20,color=(30,40,55)):
    draw.text(position,text,font=FONTS[size],fill=color)


def render_camera(env,target,distance,pitch,width,height):
    view=p.computeViewMatrixFromYawPitchRoll(target,distance,45,pitch,0,2)
    projection=p.computeProjectionMatrixFOV(55,width/height,.005,100)
    _,_,rgba,_,_=p.getCameraImage(width,height,view,projection,
        renderer=p.ER_TINY_RENDERER,physicsClientId=env.client)
    img=Image.fromarray(np.asarray(rgba,np.uint8).reshape(height,width,4)[...,:3])
    return overlay(img,env.center_estimate,view,projection),view,projection


def project(position,view,projection,width,height):
    matrix=np.asarray(projection).reshape(4,4,order='F')@np.asarray(view).reshape(4,4,order='F')
    q=matrix@np.r_[position,1.]
    if q[3]<=0 or not np.all(np.abs(q[:3])<=q[3]): return None
    return ((q[0]/q[3]+1)*width/2,(1-q[1]/q[3])*height/2)


def flight_frame(env,info,episode,seeds,steps,trail):
    canvas=background(); draw=ImageDraw.Draw(canvas)
    draw.text((16,10),'最新模型测试：无人机自主降落到变速移动平台',font=TITLE,fill=(20,35,55))
    label(draw,(16,52),f'{steps:,}步模型 | 冻结测试，无探索噪声、无参数更新 | 第{episode}/{len(seeds)}回合，种子{seeds[episode-1]}',18)
    label(draw,(16,82),'真实物理仿真全景 · 蓝线为已飞行轨迹',18)
    label(draw,(856,82),'无人机模型近景（外部观察视角）',18)
    target=(env.state[:3]+env.platform_position+np.array([0,0,.04]))*.5
    separation=float(np.linalg.norm(env.state[:3]-env.platform_position))
    distance=max(2.1,2.1+separation*.75)
    scene,view,projection=render_camera(env,target,distance,-30,824,458)
    sd=ImageDraw.Draw(scene)
    line=[project(point,view,projection,824,458) for point in trail]
    for a,b in zip(line,line[1:]):
        if a is not None and b is not None: sd.line([a,b],fill=(30,100,225),width=3)
    for xyz,name,color in [(env.state[:3],'无人机',(20,70,180)),(env.platform_position+[0,0,.04],'移动平台',(30,40,55))]:
        point=project(xyz,view,projection,824,458)
        if point:
            x,y=point; sd.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
            sd.text((x+7,y-25),name,font=FONTS[18],fill=color)
    canvas.paste(scene,(16,108))
    drone,_,_=render_camera(env,env.state[:3],.32,-25,408,210)
    canvas.paste(drone,(856,108))
    label(draw,(856,326),'机载相机图像（标线仅用于展示）',18)
    onboard=Image.fromarray(env.rgb).resize((208,208),Image.Resampling.NEAREST)
    camera_projection=p.computeProjectionMatrixFOV(env.cfg.camera_fov,1,env.cfg.camera_near,100)
    onboard=overlay(onboard,env.center_estimate,env.camera_view,camera_projection)
    canvas.paste(onboard,(856,356))
    center=env.center_estimate; color=(10,140,65) if center['mode']=='observed' else (210,115,0)
    label(draw,(1072,365),'中心估计',18)
    label(draw,(1072,397),SOURCE.get(center.get('estimate_kind'),'未知'),18,color)
    label(draw,(1072,429),f'置信度 {center.get("confidence",0):.2f}',18)
    label(draw,(1072,461),f'完整漏检',18)
    label(draw,(1072,491),f'{center.get("age_seconds") or 0:.2f} 秒',18)
    label(draw,(16,580),f'仿真时间 {env.elapsed:.2f}s  |  平台速度 {np.linalg.norm(env.platform_velocity[:2]):.2f}m/s  |  参考点距平台 {env.state[2]-env.cfg.platform_top:.2f}m',22)
    xy=float(np.linalg.norm(env.state[:2]-env.platform_position[:2]))
    event=info.get('event','running'); status=EVENT.get(event,event)
    if event=='success' and info.get('soft_landing'): status+='（软着陆）'
    label(draw,(16,619),f'评估水平误差 {xy:.3f}m  |  状态：{status}',22,(10,130,60) if event=='success' else (30,40,55))
    label(draw,(16,659),'绿色竖线：完整检测中心；橙色竖线：跟踪/预测中心。策略为视觉网络与辅助控制器的结合。',18)
    label(draw,(16,689),'视频按仿真时间1倍速输出；外部相机及误差真值仅用于展示和评估。',16)
    return canvas


def card(title,lines):
    canvas=background(); draw=ImageDraw.Draw(canvas)
    draw.text((55,80),title,font=RESULT,fill=(20,35,55))
    y=175
    for line in lines:
        label(draw,(55,y),line,22); y+=52
    return canvas


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101001,101002,101003])
    args=parser.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    agent,metadata=DrQV2.load(args.checkpoint); torch.set_num_threads(1)
    for network in (agent.encoder,agent.actor,agent.critic,agent.target): network.eval()
    config=EnvConfig(**metadata['env_config']); updates=agent.updates
    # Use exactly the saved environment/controller configuration. No easier task.
    env=MovingPlatformLanding(config,gui=False); fps=15
    video=args.output/'latest_model_test.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=fps,codec='libx264',
        quality=8,macro_block_size=16,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
        output_params=['-movflags','+faststart','-threads','2'])
    writer.send(None); frame_count=0; rows=[]
    def emit(img,count=1):
        nonlocal frame_count
        frame=np.ascontiguousarray(np.asarray(img))
        for _ in range(count): writer.send(frame); frame_count+=1
    try:
        intro=card('最新模型冻结测试',[f'检查点训练步数：{metadata["steps"]:,}；平台：变速直线运动',
            f'保存配置速度范围：{config.platform_speed_min:.2f}–{config.platform_speed_max:.2f} m/s',
            '预先固定3个展示种子，全部展示，不筛选成功回合。',
            '没有探索噪声；不更新参数；实际接触平台才判定成功。',
            '实现：DrQ-v2 + 示范辅助 + 平滑约束 + 视觉预测与近距离控制',
            '参考论文：Pawel Ladosz 等，IEEE RA-L 2024',PAPER])
        # English title uses a smaller font to remain within the canvas.
        ImageDraw.Draw(intro).rectangle((45,475,1270,545),fill=(245,247,250))
        label(ImageDraw.Draw(intro),(55,490),PAPER,18)
        emit(intro,fps*3)
        for index,seed in enumerate(args.seeds,1):
            obs,_=env.reset(seed=seed); flight=FlightLog(env); trail=[env.state[:3].copy()]
            info=dict(event='running'); next_frame=0.; ret=0.
            while True:
                action=agent.act(obs,evaluation=True)
                obs,reward,term,trunc,info=env.step(action); ret+=reward
                flight.add(env,action,info)
                if env.elapsed+1e-9>=next_frame or term or trunc:
                    trail.append(env.state[:3].copy())
                    frame=flight_frame(env,info,index,args.seeds,metadata['steps'],trail)
                    emit(frame); next_frame+=1/fps
                    if frame_count%75==0: print('VIDEO_PROGRESS',index,round(env.elapsed,2),flush=True)
                if term or trunc: break
            emit(frame,fps*2)
            frame.save(args.output/f'episode_{index:02d}_final.png')
            flight.save(args.output/f'seed_{seed}.csv')
            row=dict(seed=seed,episode=index,episode_return=ret,**flight.summary(info)); rows.append(row)
            print('VIDEO_TEST_RESULT',json.dumps(row),flush=True)
        successes=sum(row['success'] for row in rows); soft=sum(row['soft_landing'] for row in rows)
        lines=[f'本次展示测试：{successes}/{len(rows)} 次降落成功，{soft}/{len(rows)} 次软着陆。']
        for row in rows:
            lines.append(f'回合{row["episode"]}：{EVENT.get(row["event"],row["event"])} | {row["time_s"]:.2f}s | 水平误差{row["xy_error_m"]:.3f}m')
        lines+=['仅为3个固定种子的展示测试，不代表正式统计成功率。',
                '包含人工辅助估计与控制，不是论文纯 DrQ-v2 的完整复现。']
        summary=card('本次测试结果',lines); emit(summary,fps*4)
        summary.save(args.output/'summary.png')
    finally:
        writer.close(); env.close()
    assert agent.updates==updates and checksum==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result=dict(checkpoint=str(args.checkpoint),checkpoint_steps=metadata['steps'],sha256=checksum,
        environment=config.to_dict(),video=str(video),fps=fps,frames=frame_count,duration_seconds=frame_count/fps,
        simulation_playback_speed=1,optimization_updates=0,exploration_noise=False,seeds=args.seeds,
        successes=successes,soft_landings=soft,rows=rows,
        note='All 3 preselected demonstration seeds included. Not a formal success-rate estimate; hybrid controller, not pure paper reproduction.')
    (args.output/'test_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VIDEO_DONE',json.dumps(dict(video=str(video),steps=metadata['steps'],successes=successes,
        episodes=len(rows),duration_seconds=frame_count/fps)),flush=True)


if __name__=='__main__': main()