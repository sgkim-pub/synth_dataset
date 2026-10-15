"""COCO 2017 배경 사진 준비 (hw2_repro 의 download_coco2017_backgrounds.py 방식).

  1) images.cocodataset.org 에서 이미지 zip(<split>.zip) 과 annotations_trainval2017.zip 을 내려받는다
     (중간에 끊기면 이어받기, 이미 받은 zip 은 재사용).
  2) 어노테이션의 license 필드로 재배포가 쉬운 CC BY / CC BY-SA 사진만 고른다 (--licenses all 이면 전부).
  3) 고정 seed(--pool-seed)로 섞어 서로 겹치지 않는 train / val 배경 풀을 만든다
     -> 학습 이미지와 검증 이미지가 같은 배경 사진을 쓰지 않는다.
  4) <output>/pools/{train,val}/*.jpg 와 출처·라이선스를 기록한 <output>/pools/backgrounds-manifest.json 생성.

사용 예
  python coco_backgrounds.py                                  # val2017(약 1 GB) -> CC BY/BY-SA 약 1270장
  python coco_backgrounds.py --split train2017                # train2017(약 18 GB) -> 배경 수만 장
  python coco_backgrounds.py --licenses all --val-fraction 0.2

COCO 라이선스 id: 1 BY-NC-SA, 2 BY-NC, 3 BY-NC-ND, 4 BY, 5 BY-SA, 6 BY-ND, 7 저작권 없음, 8 미국 정부 저작물
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "http://images.cocodataset.org"
ANN_ZIP = "annotations/annotations_trainval2017.zip"
HERE = Path(__file__).resolve().parent


def download(url, dest):
    """이어받기를 지원하는 단순 다운로더."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.is_file():
        print(f"[재사용] {dest}")
        return dest
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:      # 서버가 이어받기를 지원하지 않으면 처음부터
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        t0, last = time.time(), 0.0
        with open(part, mode) as f:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if time.time() - last > 2:
                    last = time.time()
                    mb = have / 1e6
                    print(f"\r  {dest.name}: {mb:,.0f} / {total / 1e6:,.0f} MB  ({mb / max(1e-3, last - t0):.1f} MB/s)", end="", flush=True)
    print()
    part.rename(dest)
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val2017", "train2017"], default="val2017", help="배경으로 쓸 COCO 이미지 묶음")
    p.add_argument("--output", type=Path, default=HERE / "backgrounds/coco2017", help="배경 풀 출력 폴더")
    p.add_argument("--cache", type=Path, default=HERE / "backgrounds/downloads", help="zip 다운로드 캐시")
    p.add_argument("--licenses", default="4,5", help="허용 라이선스 id (기본 4,5 = CC BY, CC BY-SA) 또는 all")
    p.add_argument("--val-fraction", type=float, default=0.15, help="val 배경 풀 비율 (나머지는 train 풀)")
    p.add_argument("--pool-seed", type=int, default=20261013, help="풀 나누기 seed (같으면 항상 같은 풀)")
    p.add_argument("--min-side", type=int, default=320, help="짧은 변이 이보다 작은 사진 제외")
    p.add_argument("--max-images", type=int, default=0, help="사용할 최대 사진 수 (0 = 제한 없음)")
    args = p.parse_args()

    img_zip = download(f"{BASE}/zips/{args.split}.zip", args.cache / f"{args.split}.zip")
    ann_zip = download(f"{BASE}/{ANN_ZIP}", args.cache / "annotations_trainval2017.zip")

    # captions_*.json 은 instances_*.json 보다 작지만 같은 images/licenses 정보를 가진다
    with zipfile.ZipFile(ann_zip) as z:
        info = json.loads(z.read(f"annotations/captions_{args.split}.json"))
    licenses = {l["id"]: l for l in info["licenses"]}
    allowed = set(licenses) if args.licenses == "all" else {int(x) for x in args.licenses.split(",")}
    images = sorted((im for im in info["images"]
                     if im["license"] in allowed and min(im["width"], im["height"]) >= args.min_side),
                    key=lambda im: im["id"])
    random.Random(args.pool_seed).shuffle(images)
    if args.max_images:
        images = images[:args.max_images]
    n_val = max(1, round(len(images) * args.val_fraction))
    pools = {"val": images[:n_val], "train": images[n_val:]}
    print(f"{args.split}: 전체 {len(info['images'])}장 중 라이선스 {sorted(allowed)} + 짧은 변 >= {args.min_side}: "
          f"{len(images)}장 -> train {len(pools['train'])} / val {len(pools['val'])}")

    out = args.output / "pools"
    for split in pools:
        shutil.rmtree(out / split, ignore_errors=True)
        (out / split).mkdir(parents=True)
    records = []
    with zipfile.ZipFile(img_zip) as z:
        for split, ims in pools.items():
            for k, im in enumerate(ims):
                data = z.read(f"{args.split}/{im['file_name']}")
                (out / split / im["file_name"]).write_bytes(data)
                lic = licenses[im["license"]]
                records.append({"file_name": im["file_name"], "pool": split, "coco_id": im["id"],
                                "width": im["width"], "height": im["height"],
                                "license_id": im["license"], "license": lic["name"], "license_url": lic["url"],
                                "flickr_url": im.get("flickr_url"), "coco_url": im.get("coco_url"),
                                "sha256": hashlib.sha256(data).hexdigest()})
                if (k + 1) % 500 == 0:
                    print(f"  {split}: {k + 1}/{len(ims)}", flush=True)
    manifest = {"source": f"COCO 2017 {args.split}", "licenses_allowed": sorted(allowed), "pool_seed": args.pool_seed,
                "val_fraction": args.val_fraction, "min_side": args.min_side,
                "counts": {s: len(v) for s, v in pools.items()}, "images": records}
    (out / "backgrounds-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"완료: {out}  (train {len(pools['train'])}, val {len(pools['val'])})")
    print("합성 이미지를 공유할 때는 backgrounds-manifest.json 의 출처/라이선스 고지를 함께 유지하세요.")


if __name__ == "__main__":
    sys.exit(main())
