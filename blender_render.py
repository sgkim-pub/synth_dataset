"""Blender 5.2 CLI 렌더러: Wavefront OBJ 물체 + COCO17 배경 합성 -> YOLO 이미지/라벨.

make_dataset.py 가 아래처럼 실행한다 (직접 실행할 필요 없음).
    blender -b --factory-startup -noaudio -P blender_render.py -- --job <job.json>

job.json 형식
    {"settings": {...렌더/무작위화 설정...},
     "classes":  [{"id": 0, "name": "horse", "obj": "/abs/horse.obj"}, ...],
     "scenes":   [{"uid": 123, "seed": 20261015123, "class_id": 0 | -1(배경만),
                   "split": "train", "image": "/abs/.../x.jpg", "label": "/abs/.../x.txt"}, ...],
     "meta":     "/abs/.../_meta/chunk_0000.jsonl"}

장면 하나의 처리 순서 (모두 Blender 안에서 진행)
    1) 물체 크기 정규화(가장 긴 변 = 1) 후 무작위 자세(세워 두고 Z 회전 / 완전 무작위 회전)
    2) 재질 무작위 (원래 MTL 재질 유지 또는 무작위 색·거칠기·금속성)
    3) 반구 카메라: 고도각·방위각·렌즈·롤 무작위, 화면 채움 비율(=물체 크기)로 거리 결정, 화면 내 위치 무작위. 반구: 물체를 위에서 둘러싼 시점들.
    4) 조명 무작위: 주광(AREA/POINT/SUN/SPOT) + 보조광 0~3개 + 역광, 색온도·세기·크기, 배경 사진 색으로 주변광
    5) Cycles 로 투명 배경 렌더(선택적으로 그림자 받는 바닥 = shadow catcher)
    6) COCO 배경 사진을 무작위 크롭/리사이즈/밝기 변경 후 알파 합성, 카메라 잡음·블러·색 변화, JPEG 저장
    7) 물체 메시 정점을 카메라로 투영해 정확한 bbox 계산 -> YOLO 라벨 "class cx cy w h" (그림자 제외)
장면 난수는 scene["seed"] 로 다시 초기화하므로 같은 job 이면 같은 장면(배경·카메라·자세·조명)이 나온다.
"""
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

# ----------------------------------------------------------------------------- 인자
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
if len(argv) != 2 or argv[0] != "--job":
    sys.exit("usage: blender -b -P blender_render.py -- --job <job.json>")
JOB = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
S = JOB["settings"]
W, H = int(S["width"]), int(S["height"])
TMP = Path(tempfile.mkdtemp(prefix="synth_render_"))


def rng_range(key):
    lo, hi = S[key]
    return random.uniform(lo, hi)


# ----------------------------------------------------------------------------- 장면 기본 설정
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    r = scene.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.film_transparent = True                       # 물체(+그림자)만 렌더하고 배경은 나중에 합성
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"  # 사진(sRGB)과 같은 톤으로 합성
    scene.view_settings.look = "None"
    cy = scene.cycles
    cy.samples = int(S["samples"])
    cy.use_adaptive_sampling = True
    cy.use_denoising = True
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transparent_max_bounces = 6, 3, 3, 8
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.seed = 0
    if S.get("threads"):
        r.threads_mode, r.threads = "FIXED", int(S["threads"])
    cy.device = "CPU"
    if S["device"].upper() == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = S["gpu_type"]
        prefs.get_devices()
        used = [d for d in prefs.devices if d.type == S["gpu_type"]]
        for d in prefs.devices:
            d.use = d in used
        if used:
            cy.device = "GPU"
        else:
            print(f"[경고] {S['gpu_type']} GPU 를 찾지 못해 CPU 로 렌더합니다.", flush=True)
    world = bpy.data.worlds.new("World")
    scene.world = world
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit, cam_data.sensor_width = "AUTO", 36.0
    cam_data.clip_start, cam_data.clip_end = 0.01, 1000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # 그림자 받는 바닥 (shadow catcher): 투명 배경에 그림자만 알파로 남는다
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.context.object
    plane.name = "ShadowCatcher"
    plane.is_shadow_catcher = True
    plane.hide_render = True
    return scene, cam, plane, world


# ----------------------------------------------------------------------------- OBJ 불러오기
CLASS_OBJ = {}


def load_class_object(cls):
    """OBJ 를 한 번만 불러와 하나의 메시로 합치고, bbox 중심 = 원점, 가장 긴 변 = 1 로 정규화."""
    if cls["id"] in CLASS_OBJ:
        return CLASS_OBJ[cls["id"]]
    before = set(bpy.data.objects)
    # Wavefront OBJ 기본 규약(파일 안에서 +Y 가 위) -> Blender Z-up 으로 변환
    bpy.ops.wm.obj_import(filepath=cls["obj"], forward_axis=S["obj_forward"], up_axis=S["obj_up"])
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for o in set(bpy.data.objects) - before - set(objs):
        bpy.data.objects.remove(o, do_unlink=True)
    if not objs:
        raise RuntimeError(f"OBJ 에 메시가 없습니다: {cls['obj']}")
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=objs[0], object=objs[0], selected_objects=objs,
                                       selected_editable_objects=objs):
            bpy.ops.object.join()
    obj = objs[0]
    obj.name = f"target_{cls['name']}"
    mesh = obj.data
    mesh.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    co = mesh_vertices(mesh)
    lo, hi = co.min(axis=0), co.max(axis=0)
    size = float((hi - lo).max())
    if size <= 0:
        raise RuntimeError(f"크기가 0 인 OBJ: {cls['obj']}")
    mesh.transform(Matrix.Scale(1.0 / size, 4) @ Matrix.Translation(Vector(-(lo + hi) / 2)))
    mesh.update()
    orig = [m for m in mesh.materials]
    rand_mat = make_random_material(f"rand_{cls['name']}")
    if not orig:
        mesh.materials.append(rand_mat)
    obj.hide_render = True
    CLASS_OBJ[cls["id"]] = {"obj": obj, "orig": orig, "rand": rand_mat,
                            "verts": mesh_vertices(mesh), "bottom": float(mesh_vertices(mesh)[:, 2].min())}
    print(f"loaded {cls['name']}: {len(mesh.vertices)} verts, {len(mesh.polygons)} faces, "
          f"materials={len(orig)} (원래 크기 {size:.4g})", flush=True)
    return CLASS_OBJ[cls["id"]]


def mesh_vertices(mesh):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def make_random_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    return mat


def randomize_material(entry):
    """원래 재질(MTL) 유지 또는 무작위 단색 PBR 재질."""
    mesh = entry["obj"].data
    mode = S["material"]
    use_orig = bool(entry["orig"]) and (mode == "original" or (mode == "mixed" and random.random() < S["keep_material_prob"]))
    if use_orig:
        for i, m in enumerate(entry["orig"]):
            mesh.materials[i] = m
        return "original"
    bsdf = next(n for n in entry["rand"].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    h, s, v = random.random(), random.uniform(0.0, 1.0), random.uniform(0.05, 0.95)
    rgb = hsv_to_linear(h, s, v)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = random.uniform(0.15, 0.95)
    bsdf.inputs["Metallic"].default_value = random.uniform(0.3, 1.0) if random.random() < 0.15 else 0.0
    bsdf.inputs["Coat Weight"].default_value = random.uniform(0.0, 0.6) if random.random() < 0.2 else 0.0
    for i in range(len(mesh.materials)):
        mesh.materials[i] = entry["rand"]
    return {"hsv": [round(h, 3), round(s, 3), round(v, 3)]}


def hsv_to_linear(h, s, v):
    import colorsys
    return [c ** 2.2 for c in colorsys.hsv_to_rgb(h, s, v)]  # sRGB 감마 -> Blender 선형 색


def kelvin_to_rgb(k):
    """색온도(K) -> 대략적인 RGB (Tanner Helland 근사), 0~1."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return [min(1.0, max(0.0, c / 255.0)) for c in (r, g, b)]


# ----------------------------------------------------------------------------- 자세 / 카메라 / 조명
def random_rotation():
    """배치 각도: upright_prob 확률로 세워 두고(Z 회전 + 작은 기울기), 아니면 SO(3) 균일 무작위 회전."""
    if random.random() < S["upright_prob"]:
        tilt = math.radians(S["upright_tilt"])
        rot = (Matrix.Rotation(random.uniform(0, 2 * math.pi), 4, "Z")
               @ Matrix.Rotation(random.uniform(-tilt, tilt), 4, "X")
               @ Matrix.Rotation(random.uniform(-tilt, tilt), 4, "Y"))
        return rot, "upright"
    u1, u2, u3 = random.random(), random.random(), random.random()  # 균일 쿼터니언 (Shoemake)
    q = (math.sqrt(1 - u1) * math.sin(2 * math.pi * u2), math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
         math.sqrt(u1) * math.sin(2 * math.pi * u3), math.sqrt(u1) * math.cos(2 * math.pi * u3))
    from mathutils import Quaternion
    return Quaternion((q[3], q[0], q[1], q[2])).to_matrix().to_4x4(), "random"


def place_camera(cam, centre, radius):
    """물체 경계 구(centre, radius)를 둘러싼 반구 위에 카메라를 두고, 화면 채움 비율로 거리를 정한다.
    화면 안 위치(중앙/가장자리)는 남는 화각 안에서 무작위로 흔든다."""
    lens = rng_range("cam_lens")
    cam.data.lens = lens
    half_h = math.atan(18.0 / lens)                                      # 가로 반화각 (36 mm 센서, 가로 기준)
    half_v = math.atan(math.tan(half_h) * min(W, H) / max(W, H))         # 세로(짧은 변) 반화각
    if H > W:
        half_h, half_v = half_v, half_h
    fill = rng_range("size_range")                                       # 물체 경계 구 지름 / 짧은 변 화각
    r = radius / math.sin(min(fill, 1.0) * min(half_h, half_v))
    elev = math.radians(rng_range("cam_elev"))
    az = random.uniform(0, 2 * math.pi)
    pos = centre + r * Vector((math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)))
    forward = (centre - pos).normalized()
    right = forward.cross(Vector((0, 0, 1)))
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    right.normalize()
    up = right.cross(forward)
    ang = math.asin(min(1.0, radius / r))
    sx = max(0.0, half_h - ang) * S["offcenter"]
    sy = max(0.0, min(half_h, half_v) - ang) * S["offcenter"]
    target = centre + right * r * math.tan(random.uniform(-sx, sx)) + up * r * math.tan(random.uniform(-sy, sy))
    look = (target - pos).normalized()
    rot = look.to_track_quat("-Z", "Y")
    roll = math.radians(rng_range("cam_roll"))
    cam.matrix_world = Matrix.Translation(pos) @ (rot.to_matrix().to_4x4() @ Matrix.Rotation(roll, 4, "Z"))
    return {"lens_mm": round(lens, 2), "fill": round(fill, 3), "dist": round(r, 3), "elev_deg": round(math.degrees(elev), 1),
            "azim_deg": round(math.degrees(az), 1), "roll_deg": round(math.degrees(roll), 1)}


LIGHTS = []


def clear_lights():
    for ob in LIGHTS:
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.lights.remove(data)
    LIGHTS.clear()


def add_light(kind, centre, dist, elev_deg, az_deg, power, kelvin, size):
    data = bpy.data.lights.new(f"L{len(LIGHTS)}", kind)
    data.color = kelvin_to_rgb(kelvin)
    if kind == "SUN":
        data.energy = power
        data.angle = math.radians(size)
    else:
        data.energy = power * dist * dist           # 거리 제곱 보정 -> 물체가 받는 조도는 power 에 비례
        if kind == "AREA":
            data.size = size * dist
        else:
            data.shadow_soft_size = size * dist
        if kind == "SPOT":
            data.spot_size = math.radians(random.uniform(35, 80))
            data.spot_blend = random.uniform(0.1, 0.6)
    ob = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(ob)
    e, a = math.radians(elev_deg), math.radians(az_deg)
    pos = centre + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    ob.matrix_world = Matrix.Translation(pos) @ (centre - pos).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    LIGHTS.append(ob)
    return {"type": kind, "elev": round(elev_deg, 1), "azim": round(az_deg, 1), "power": round(power, 2), "K": int(kelvin)}


def randomize_lighting(world, centre, cam_azim_deg, bg_mean):
    """조명: 주광 1개 + 보조광 0~3개 + (확률) 역광 + 배경 사진 평균색의 주변광."""
    clear_lights()
    gain = rng_range("light_gain")
    info = []
    kind = random.choices(["AREA", "POINT", "SUN", "SPOT"], weights=[4, 2, 2, 1])[0]
    key_power = (random.uniform(1.5, 5.0) if kind == "SUN" else random.uniform(15, 70)) * gain
    info.append(add_light(kind, centre, random.uniform(3, 8), random.uniform(10, 85),
                          cam_azim_deg + random.uniform(-120, 120), key_power,
                          random.uniform(2700, 7500), random.uniform(1, 15) if kind == "SUN" else random.uniform(0.05, 0.8)))
    for _ in range(random.randint(0, 3)):
        fk = random.choice(["AREA", "POINT"])
        info.append(add_light(fk, centre, random.uniform(3, 8), random.uniform(-10, 70), random.uniform(0, 360),
                              random.uniform(2, 25) * gain, random.uniform(2700, 9000), random.uniform(0.1, 1.0)))
    if random.random() < 0.3:  # 역광(림 라이트)
        info.append(add_light("AREA", centre, random.uniform(3, 6), random.uniform(10, 60), cam_azim_deg + 180 + random.uniform(-40, 40),
                              random.uniform(10, 50) * gain, random.uniform(3500, 9000), random.uniform(0.1, 0.5)))
    # 주변광: 배경 사진의 평균색(선형)을 약하게 -> 물체 색감이 배경과 어울리게
    amb = rng_range("ambient")
    tint = [max(0.02, c) ** 2.2 for c in bg_mean]
    bgnode = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bgnode.inputs["Color"].default_value = (*tint, 1.0)
    bgnode.inputs["Strength"].default_value = amb / max(1e-3, float(np.mean(tint))) * 0.25
    return {"lights": info, "ambient": round(amb, 3), "gain": round(gain, 3)}


# ----------------------------------------------------------------------------- 투영 bbox
def projected_bbox(obj, cam, verts):
    """메시 정점을 카메라로 투영한 정확한 bbox (정규화 xyxy, 위쪽 원점). 그림자는 포함하지 않는다."""
    M = np.array(obj.matrix_world)
    wv = verts @ M[:3, :3].T + M[:3, 3]
    V = np.array(cam.matrix_world.inverted())
    c = wv @ V[:3, :3].T + V[:3, 3]                       # 카메라 좌표 (카메라는 -Z 를 봄)
    if (c[:, 2] > -cam.data.clip_start).any():
        return None
    f = cam.data.lens / 36.0 * max(W, H)                  # 초점거리(px), sensor_fit AUTO = 긴 변 기준
    u = W / 2 + f * c[:, 0] / -c[:, 2]
    v = H / 2 - f * c[:, 1] / -c[:, 2]
    full = (u.min(), v.min(), u.max(), v.max())
    clip = (max(0.0, full[0]), max(0.0, full[1]), min(W, full[2]), min(H, full[3]))
    area_full = (full[2] - full[0]) * (full[3] - full[1])
    area_clip = max(0.0, clip[2] - clip[0]) * max(0.0, clip[3] - clip[1])
    return clip, (area_clip / area_full if area_full > 0 else 0.0)


# ----------------------------------------------------------------------------- 이미지 처리 (numpy, Blender 내장)
def load_pixels(path):
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)                            # Blender 순서: 아래쪽 행이 먼저


def resize_bilinear(img, out_w, out_h, x0, y0, cw, ch):
    """img[y0:y0+ch, x0:x0+cw] 영역을 (out_h, out_w) 로 양선형 보간."""
    ys = y0 + (np.arange(out_h) + 0.5) * ch / out_h - 0.5
    xs = x0 + (np.arange(out_w) + 0.5) * cw / out_w - 0.5
    ys = np.clip(ys, 0, img.shape[0] - 1); xs = np.clip(xs, 0, img.shape[1] - 1)
    y0i = np.floor(ys).astype(int); x0i = np.floor(xs).astype(int)
    y1i = np.minimum(y0i + 1, img.shape[0] - 1); x1i = np.minimum(x0i + 1, img.shape[1] - 1)
    wy = (ys - y0i)[:, None, None]; wx = (xs - x0i)[None, :, None]
    a = img[y0i][:, x0i]; b = img[y0i][:, x1i]; c = img[y1i][:, x0i]; d = img[y1i][:, x1i]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def gaussian_blur(img, sigma):
    k = max(1, int(round(sigma * 2.5)))
    x = np.arange(-k, k + 1)
    w = np.exp(-x * x / (2 * sigma * sigma)); w /= w.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim; pad[axis] = (k, k)
        p = np.pad(img, pad, mode="edge")
        n = img.shape[axis]
        img = sum(wi * np.take(p, np.arange(i, i + n), axis=axis) for i, wi in enumerate(w))
    return img


BG_PATHS = {}


def choose_background(split):
    pool = Path(S["backgrounds"]) / split
    if split not in BG_PATHS:
        BG_PATHS[split] = sorted(p for p in pool.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not BG_PATHS[split]:
            raise FileNotFoundError(f"배경 사진이 없습니다: {pool}")
    path = random.choice(BG_PATHS[split])
    src = load_pixels(path)[..., :3]
    sh, sw = src.shape[:2]
    # 무작위 크롭(출력 비율 유지, 최대 크기의 zoom 배) + 좌우 뒤집기
    zoom = random.uniform(*S["bg_zoom"])
    cw = min(sw, sh * W / H) * zoom
    ch = cw * H / W
    x0, y0 = random.uniform(0, sw - cw), random.uniform(0, sh - ch)
    bg = resize_bilinear(src, W, H, x0, y0, cw, ch)
    if random.random() < 0.5:
        bg = bg[:, ::-1]
    if random.random() < 0.3:
        bg = gaussian_blur(bg, random.uniform(0.5, 2.0))  # 초점 흐림(피사계 심도 느낌)
    if random.random() < 0.5:
        bg = bg * random.uniform(0.75, 1.15) + random.uniform(-0.06, 0.04)
    return np.clip(bg, 0, 1), path, {"crop": [round(x0), round(y0), round(cw), round(ch)], "zoom": round(zoom, 3)}


def camera_effects(img):
    """합성 후 전체 이미지에 같은 카메라 효과: 색/감마 변화, 블러, 센서 잡음."""
    fx = {}
    if random.random() < 0.6:
        g = np.array([random.uniform(0.92, 1.08) for _ in range(3)], dtype=np.float32)
        gamma = random.uniform(0.85, 1.15)
        img = np.clip(img * g, 0, 1) ** gamma
        fx["color"] = [round(float(x), 3) for x in g] + [round(gamma, 3)]
    if random.random() < S["blur_prob"]:
        s = random.uniform(0.4, 1.3)
        img = gaussian_blur(img, s)
        fx["blur"] = round(s, 2)
    if random.random() < S["noise_prob"]:
        s = random.uniform(0.004, 0.03)
        img = img + np.random.normal(0, s, img.shape).astype(np.float32)
        fx["noise"] = round(s, 4)
    return np.clip(img, 0, 1), fx


def save_jpeg(img, path, quality):
    h, w = img.shape[:2]
    out = bpy.data.images.new("out", w, h, alpha=False, float_buffer=False)
    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2)
    out.pixels.foreach_set(rgba.astype(np.float32).ravel())
    out.file_format = "JPEG"
    tmp = Path(str(path) + ".tmp.jpg")
    out.save(filepath=str(tmp), quality=int(quality))
    bpy.data.images.remove(out)
    os.replace(tmp, path)                                 # 끝까지 쓴 파일만 보이게 (중단 후 재개 안전)


# ----------------------------------------------------------------------------- 장면 렌더
def render_scene(sc, scene, cam, plane, world, classes):
    random.seed(sc["seed"]); np.random.seed(sc["seed"] % (2 ** 32))
    for e in CLASS_OBJ.values():
        e["obj"].hide_render = True
    bg, bg_path, bg_info = choose_background(sc["split"])
    meta = {"uid": sc["uid"], "seed": sc["seed"], "split": sc["split"], "image": Path(sc["image"]).name,
            "background": bg_path.name, "bg": bg_info}
    label = ""
    if sc["class_id"] < 0:                                # 배경만 있는 negative 장면 (라벨 비어 있음)
        img = bg
        meta["class"] = "negative"
    else:
        cls = classes[sc["class_id"]]
        entry = load_class_object(cls)
        obj = entry["obj"]
        meta["class"] = cls["name"]
        for attempt in range(40):
            rot, rot_mode = random_rotation()
            obj.matrix_world = rot
            bpy.context.view_layer.update()
            M = np.array(obj.matrix_world)
            wv = entry["verts"] @ M[:3, :3].T
            lo, hi = wv.min(axis=0), wv.max(axis=0)
            centre_np = (lo + hi) / 2
            radius = float(np.linalg.norm(wv - centre_np, axis=1).max())
            cam_info = place_camera(cam, Vector(centre_np.tolist()), radius)
            bpy.context.view_layer.update()
            res = projected_bbox(obj, cam, entry["verts"])
            if res is None:
                continue
            (x1, y1, x2, y2), vis = res
            if vis >= S["min_visible"] and min(x2 - x1, y2 - y1) >= S["min_box_px"]:
                break
        else:
            raise RuntimeError(f"장면 {sc['uid']}: 조건을 만족하는 카메라를 찾지 못했습니다")
        meta.update(rotation=rot_mode, rotation_euler=[round(math.degrees(a), 1) for a in rot.to_euler()],
                    camera=cam_info, material=randomize_material(entry))
        # 그림자 바닥: 카메라가 위에서 볼 때만 (아래에서 보면 바닥이 물체를 가림)
        use_shadow = cam_info["elev_deg"] > 3 and random.random() < S["shadow_prob"]
        plane.hide_render = not use_shadow
        if use_shadow:
            plane.matrix_world = Matrix.Translation((centre_np[0], centre_np[1], lo[2] - 1e-4)) @ Matrix.Scale(radius * 40, 4)
        meta["shadow"] = use_shadow
        bg_mean = bg.reshape(-1, 3).mean(axis=0).tolist()
        meta["lighting"] = randomize_lighting(world, Vector(centre_np.tolist()), cam_info["azim_deg"], bg_mean)
        obj.hide_render = False
        scene.cycles.seed = sc["seed"] % (2 ** 31)
        fg_path = TMP / "fg.png"
        scene.render.filepath = str(fg_path)
        bpy.ops.render.render(write_still=True)
        fg = load_pixels(fg_path)
        a = fg[..., 3:4]
        img = fg[..., :3] * a + bg * (1 - a)              # 알파 합성 (그림자는 알파만 있는 어두운 영역)
        # YOLO bbox: Blender 행 순서(아래->위)와 무관하게 위쪽 원점 픽셀 좌표로 계산됨
        cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
        label = f"{sc['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        meta["bbox_xyxy_px"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        meta["visible_ratio"] = round(vis, 3)
    img, meta["fx"] = camera_effects(img)
    q = random.randint(*S["jpeg_quality"])
    meta["jpeg_quality"] = q
    Path(sc["label"]).write_text(label, encoding="utf-8")
    save_jpeg(img, sc["image"], q)                        # 이미지를 마지막에 저장 -> 이미지가 있으면 라벨도 있음
    return meta


def main():
    scene, cam, plane, world = setup_scene()
    classes = {c["id"]: c for c in JOB["classes"]}
    meta_path = Path(JOB["meta"])
    t0 = time.time()
    done = 0
    for sc in JOB["scenes"]:
        if S.get("resume") and Path(sc["image"]).is_file() and Path(sc["label"]).is_file():
            continue
        t = time.time()
        meta = render_scene(sc, scene, cam, plane, world, classes)
        meta["seconds"] = round(time.time() - t, 2)
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        done += 1
        print(f"[{done}/{len(JOB['scenes'])}] {Path(sc['image']).name} {meta['class']} {meta['seconds']}s", flush=True)
    print(f"chunk done: {done} scenes in {time.time() - t0:.0f}s", flush=True)


main()
