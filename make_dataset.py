"""YOLO11n 학습용 합성 데이터셋 생성기 (Wavefront OBJ 물체 + COCO17 배경, Blender 5.2 CLI 렌더).

구성 (파이썬 스크립트 3개)
  coco_backgrounds.py : COCO 2017 배경 사진 다운로드 + 겹치지 않는 train/val 배경 풀 생성.
  make_dataset.py     : (이 파일) 장면 목록 작성 -> Blender 여러 개 병렬 실행 -> 검사 -> datasets.yaml 생성.
  blender_render.py   : Blender 안에서 실행되는 렌더러 (물체 배치·카메라·조명 무작위화, 배경 합성, 레이블링 등)

사용 예
  python coco_backgrounds.py                                       # 처음 한 번 (배경 파일 준비)
  python make_dataset.py --objects objs/                           # objs/*.obj 각 500장 + val 각 100장
  python make_dataset.py --objects objs/ -n 20 --output smoke      # Smoke 테스트. 동작 확인
  python make_dataset.py --objects horse.obj hydra.obj -n 1000 --blender ~/blender-5.2.0-linux-x64/blender

결과 (기본 --output datasets)
  datasets/images/train/*.jpg   datasets/labels/train/*.txt   (클래스당 n 장)
  datasets/images/val/*.jpg     datasets/labels/val/*.txt     (클래스당 n 장의 20%, train set과 겹치지 않게 별도 생성)
  datasets/datasets.yaml        Ultralytics YOLO모델 훈련을 위한 데이터셋 위치 기술(train, val paths)
  datasets/_meta/*.json         장면별 기록 (배경 사진, 카메라, 자세, 조명, bbox 등)
  datasets/_logs/*.log          Blender 출력 로그
레이블 형식: 한 줄에 "class_id cx cy w h" (0~1 정규화, YOLO bbox). 클래스 id 는 --objects 순서(폴더면 파일 이름 정렬순).

학습 예: yolo detect train model=yolo11n.pt data=datasets/datasets.yaml imgsz=640 epochs=100
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RENDERER = HERE / "blender_render.py"
EXTS = (".jpg",)


def pair(text, cast=float):
    a, b = (cast(x) for x in text.split(","))
    if a > b:
        raise argparse.ArgumentTypeError(f"범위가 거꾸로입니다: {text}")
    return [a, b]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("입력/출력")
    g.add_argument("--objects", nargs="+", required=True, type=Path,
                   help="OBJ 파일들 또는 OBJ 가 든 폴더. 파일 이름(확장자 제외)이 클래스 이름")
    g.add_argument("--names", default="", help="클래스 이름 직접 지정 (쉼표 구분, --objects 순서와 같게)")
    g.add_argument("--output", type=Path, default=HERE / "datasets", help="데이터셋 폴더 (기본 ./datasets)")
    g.add_argument("--backgrounds", type=Path, default=HERE / "backgrounds/coco2017/pools",
                   help="coco_backgrounds.py 가 만든 배경 풀 폴더 (train/, val/ 포함)")
    g.add_argument("-n", "--num-images", type=int, default=500, help="물체(클래스)별 train 이미지 수 (기본 500)")
    g.add_argument("--val-ratio", type=float, default=0.2, help="val 이미지 수 = N x 이 비율 (기본 0.2, train 과 별도 생성)")
    g.add_argument("--negatives", type=float, default=0.0,
                   help="물체 없는 배경 이미지 비율 (전체 물체 이미지 수 대비, 예: 0.05). 기본 0 = 생성 안 함")
    g.add_argument("--seed", type=int, default=20261015, help="장면 seed = seed + 장면 번호 (같으면 같은 데이터셋)")
    g.add_argument("--resume", action="store_true", help="기존 출력 유지, 이미 만든 이미지는 건너뛰고 이어서 생성")
    g.add_argument("--overwrite", action="store_true", help="기존 출력 폴더를 지우고 새로 생성")

    g = p.add_argument_group("Blender / 렌더")
    g.add_argument("--blender", default=os.environ.get("BLENDER", "blender"), help="Blender 5.2 실행 파일 (환경변수 BLENDER)")
    g.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 4)), help="동시에 돌릴 Blender 프로세스 수")
    g.add_argument("--chunk", type=int, default=50, help="Blender 프로세스 하나가 렌더링할 장면 수")
    g.add_argument("--threads", type=int, default=0, help="Blender 프로세스당 CPU 스레드 (0 = 자동)")
    g.add_argument("--device", choices=["CPU", "GPU"], default="CPU", help="Cycles 렌더 장치")
    g.add_argument("--gpu-type", choices=["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"], default="OPTIX")
    g.add_argument("--samples", type=int, default=48, help="Cycles 샘플 수 (디노이즈 사용)")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--obj-up", default="Y", choices=["X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z"],
                   help="OBJ 파일의 위쪽 축 (Blender OBJ 기본 규약 Y)")
    g.add_argument("--obj-forward", default="NEGATIVE_Z", choices=["X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z"])

    g = p.add_argument_group("무작위화 범위")
    g.add_argument("--size-range", type=pair, default=[0.15, 0.95],
                   help="물체 크기: 경계 구 지름이 화면 짧은 변에서 차지하는 비율 범위")
    g.add_argument("--cam-elev", type=pair, default=[-5.0, 80.0], help="시점: 카메라 고도각 범위(도)")
    g.add_argument("--cam-lens", type=pair, default=[18.0, 60.0], help="렌즈 초점거리 범위(mm, 36 mm 센서)")
    g.add_argument("--cam-roll", type=pair, default=[-12.0, 12.0], help="카메라 롤(기울임) 범위(도)")
    g.add_argument("--offcenter", type=float, default=0.9, help="화면 안 물체 위치 흔들기 정도 (0 = 항상 중앙, 1 = 가장자리까지)")
    g.add_argument("--upright-prob", type=float, default=0.5, help="배치 각도: 세워 둔 자세(Z 회전) 확률. 나머지는 완전 무작위 회전")
    g.add_argument("--upright-tilt", type=float, default=15.0, help="세워 둔 자세의 최대 기울기(도)")
    g.add_argument("--light-gain", type=pair, default=[0.35, 1.8], help="조명 전체 세기 배율 범위")
    g.add_argument("--ambient", type=pair, default=[0.05, 0.6], help="주변광(배경 색) 세기 범위")
    g.add_argument("--shadow-prob", type=float, default=0.5, help="바닥 그림자(shadow catcher) 확률")
    g.add_argument("--material", choices=["mixed", "random", "original"], default="mixed",
                   help="재질: original=OBJ 의 MTL 재질, random=무작위 색, mixed=섞기")
    g.add_argument("--keep-material-prob", type=float, default=0.5, help="mixed 일 때 원래 재질을 쓸 확률 (MTL 이 있을 때)")
    g.add_argument("--bg-zoom", type=pair, default=[0.5, 1.0], help="배경 사진 크롭 크기 비율 범위")
    g.add_argument("--blur-prob", type=float, default=0.2)
    g.add_argument("--noise-prob", type=float, default=0.5)
    g.add_argument("--jpeg-quality", type=lambda t: pair(t, int), default=[75, 95])
    g.add_argument("--min-visible", type=float, default=0.85, help="화면 안에 보여야 하는 bbox 최소 비율")
    g.add_argument("--min-box-px", type=float, default=12.0, help="bbox 최소 변 길이(px)")
    return p.parse_args()


def collect_objects(args):
    files = []
    for path in args.objects:
        if path.is_dir():
            # 확장자 대소문자 무시 (Windows 는 glob 이 대소문자를 구분하지 않아 *.obj/*.OBJ 를 따로 찾으면 중복됨)
            files += sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".obj")
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"[오류] OBJ 를 찾을 수 없습니다: {path}")
    if not files:
        sys.exit("[오류] OBJ 파일이 없습니다.")
    names = [n.strip() for n in args.names.split(",")] if args.names else [f.stem for f in files]
    if len(names) != len(files) or len(set(names)) != len(names):
        sys.exit(f"[오류] 클래스 이름 수/중복 확인: {names}")
    for n in names:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", n):
            sys.exit(f"[오류] 클래스 이름은 영문/숫자/_/- 만 쓰세요: {n!r}")
    return [{"id": i, "name": n, "obj": str(f.resolve())} for i, (n, f) in enumerate(zip(names, files))]


def check_blender(blender):
    exe = shutil.which(blender) or (blender if Path(blender).is_file() else None)
    if not exe:
        sys.exit(f"[오류] Blender 실행 파일을 찾을 수 없습니다: {blender}\n"
                 "  Blender 5.2 를 설치하고 --blender <경로> 또는 환경변수 BLENDER 로 지정하세요.\n"
                 "  예: https://download.blender.org/release/Blender5.2/ 의 blender-5.2.x-linux-x64.tar.xz")
    out = subprocess.run([exe, "-b", "--factory-startup", "--version"], capture_output=True, text=True).stdout
    m = re.search(r"Blender (\d+)\.(\d+)", out)
    ver = f"{m.group(1)}.{m.group(2)}" if m else "?"
    if ver != "5.2":
        print(f"[경고] Blender {ver} 입니다. 이 프로그램은 Blender 5.2 기준으로 작성되었습니다.")
    return exe, ver


def build_scenes(args, classes):
    """장면 목록. 장면 번호(uid) = 클래스 블록(1,000,000 간격) + split 블록(train 0, val 500,000) + 순번.
    seed = --seed + uid 이므로 클래스 당 데이터 수인 n 을 늘려도 앞부분은 그대로다."""
    n_train = args.num_images
    n_val = round(args.num_images * args.val_ratio)
    scenes = []
    blocks = [(c["id"], c["name"]) for c in classes]
    n_neg = round(len(classes) * args.num_images * args.negatives)
    if n_neg:
        blocks.append((-1, "negative"))
    for cid, name in blocks:
        block = (cid if cid >= 0 else 999) * 1_000_000
        counts = {"train": n_train, "val": n_val} if cid >= 0 else {"train": n_neg, "val": round(n_neg * args.val_ratio)}
        for split, count in counts.items():
            for i in range(count):
                uid = block + (500_000 if split == "val" else 0) + i
                stem = f"{name}_{split}_{i:06d}"
                scenes.append({"uid": uid, "seed": args.seed + uid, "class_id": cid, "split": split,
                               "image": str(args.output / "images" / split / f"{stem}.jpg"),
                               "label": str(args.output / "labels" / split / f"{stem}.txt")})
    return scenes, n_train, n_val


def verify(args, classes, scenes):
    """모든 이미지에 레이블이 있는지, 레이블 형식/클래스/범위가 맞는지 검사하고 개수를 센다."""
    counts = {s: {c["name"]: 0 for c in classes} | {"negative": 0} for s in ("train", "val")}
    errors = []
    for sc in scenes:
        img, lab = Path(sc["image"]), Path(sc["label"])
        if not img.is_file() or not lab.is_file():
            errors.append(f"누락: {img.name}")
            continue
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        if sc["class_id"] < 0:
            ok = not rows
        else:
            ok = len(rows) == 1 and len(rows[0]) == 5 and int(rows[0][0]) == sc["class_id"]
            if ok:
                cx, cy, w, h = map(float, rows[0][1:])
                ok = w > 0 and h > 0 and cx - w / 2 >= -1e-4 and cy - h / 2 >= -1e-4 and cx + w / 2 <= 1.0001 and cy + h / 2 <= 1.0001
        if not ok:
            errors.append(f"잘못된 레이블: {lab}")
        name = classes[sc["class_id"]]["name"] if sc["class_id"] >= 0 else "negative"
        counts[sc["split"]][name] += 1
    return counts, errors


def write_yaml(args, classes):
    lines = ["# YOLO11n 학습용 합성 데이터셋 (make_dataset.py 생성)",
             f"path: {args.output.resolve()}",
             "train: images/train",
             "val: images/val",
             "",
             f"nc: {len(classes)}",
             "names:"] + [f"  {c['id']}: {c['name']}" for c in classes]
    path = args.output / "datasets.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    args.output = args.output.resolve()
    classes = collect_objects(args)
    for split in ("train", "val"):
        if not any((args.backgrounds / split).glob("*.jp*g")):
            sys.exit(f"[오류] 배경 사진이 없습니다: {args.backgrounds / split}\n  먼저 python coco_backgrounds.py 를 실행하세요.")
    exe, ver = check_blender(args.blender)

    if args.output.exists() and any(args.output.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.output)
        elif not args.resume:
            sys.exit(f"[오류] 출력 폴더가 비어 있지 않습니다: {args.output}\n  이어서 하려면 --resume, 새로 만들려면 --overwrite")
    for d in ("images/train", "images/val", "labels/train", "labels/val", "_meta", "_jobs", "_logs"):
        (args.output / d).mkdir(parents=True, exist_ok=True)

    scenes, n_train, n_val = build_scenes(args, classes)
    todo = [s for s in scenes if not (Path(s["image"]).is_file() and Path(s["label"]).is_file())]
    print(f"Blender {ver} ({exe})")
    print(f"클래스 {len(classes)}개: {[c['name'] for c in classes]}")
    print(f"클래스당 train {n_train} / val {n_val}, 전체 {len(scenes)}장 (남은 {len(todo)}장), "
          f"worker {args.workers} x chunk {args.chunk}, {args.device}")

    settings = {
        "width": args.width, "height": args.height, "samples": args.samples, "device": args.device, "gpu_type": args.gpu_type,
        "threads": args.threads, "backgrounds": str(args.backgrounds.resolve()), "resume": True,
        "obj_up": args.obj_up, "obj_forward": args.obj_forward,
        "size_range": args.size_range, "cam_elev": args.cam_elev, "cam_lens": args.cam_lens, "cam_roll": args.cam_roll,
        "offcenter": args.offcenter, "upright_prob": args.upright_prob, "upright_tilt": args.upright_tilt,
        "light_gain": args.light_gain, "ambient": args.ambient, "shadow_prob": args.shadow_prob,
        "material": args.material, "keep_material_prob": args.keep_material_prob, "bg_zoom": args.bg_zoom,
        "blur_prob": args.blur_prob, "noise_prob": args.noise_prob, "jpeg_quality": args.jpeg_quality,
        "min_visible": args.min_visible, "min_box_px": args.min_box_px,
    }
    (args.output / "_meta/settings.json").write_text(json.dumps(
        {"settings": settings, "classes": classes, "seed": args.seed, "num_images": n_train, "val_images": n_val,
         "blender": ver}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jobs = []
    for k in range(0, len(todo), args.chunk):
        part = todo[k:k + args.chunk]
        tag = f"chunk_{part[0]['uid']:09d}"
        job = {"settings": settings, "classes": classes, "scenes": part, "meta": str(args.output / "_meta" / f"{tag}.jsonl")}
        path = args.output / "_jobs" / f"{tag}.json"
        path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        jobs.append((tag, path, len(part)))

    started = time.time()
    done = 0

    def run(job):
        tag, path, n = job
        cmd = [exe, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
               "-P", str(RENDERER), "--", "--job", str(path)]
        with open(args.output / "_logs" / f"{tag}.log", "a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        return tag, n, rc

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            tag, n, rc = fut.result()
            if rc:
                failed.append(tag)
                print(f"[실패] {tag} (로그: {args.output / '_logs' / (tag + '.log')})", flush=True)
                continue
            done += n
            el = time.time() - started
            eta = el / done * (len(todo) - done) if done else 0
            print(f"[{done}/{len(todo)}] {tag} 완료  경과 {el / 60:.1f}분, 남은 예상 {eta / 60:.1f}분", flush=True)

    counts, errors = verify(args, classes, scenes)
    yaml_path = write_yaml(args, classes)
    print(json.dumps(counts, ensure_ascii=False, indent=1))
    print(f"datasets.yaml: {yaml_path}")
    if failed or errors:
        print(f"[경고] 실패 청크 {len(failed)}개, 문제 {len(errors)}건 (예: {errors[:5]}). --resume 으로 다시 실행하면 빠진 장면만 렌더합니다.")
        sys.exit(1)
    shutil.rmtree(args.output / "_jobs", ignore_errors=True)
    print(f"완료: {len(scenes)}장, {(time.time() - started) / 60:.1f}분")


if __name__ == "__main__":
    main()
