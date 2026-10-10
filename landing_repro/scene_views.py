"""External inspection cameras. These images never enter policy observations."""
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


def close_view(env, target, distance, pitch, width=360, height=250):
    view = p.computeViewMatrixFromYawPitchRoll(target, distance, 45, pitch, 0, 2)
    projection = p.computeProjectionMatrixFOV(55, width / height, .005, 150)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, projection,
        renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)
    return Image.fromarray(np.asarray(rgba, np.uint8).reshape(height, width, 4)[..., :3])


def inspection_views(env):
    return (close_view(env, env.state[:3], .32, -25),
            close_view(env, env.platform_position + [0, 0, .04], 1.65, -55),
            Image.fromarray(env.rgb).resize((250, 250), Image.Resampling.NEAREST))


def save_inspection(views, output, caption):
    canvas = Image.new("RGB", (970, 295), "white")
    draw = ImageDraw.Draw(canvas)
    for view, x, label in zip(views, (0, 360, 720),
                            ("Actual drone - external close view", "Actual platform - external close view", "Onboard camera - model input")):
        canvas.paste(view, (x, 22))
        draw.text((x+5, 4), label, fill="black")
    draw.text((5, 278), caption, fill="black")
    canvas.save(output)
