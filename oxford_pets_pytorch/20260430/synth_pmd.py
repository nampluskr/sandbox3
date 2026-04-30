import os
import csv
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ── 설정 ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 512
OUT_DIR    = "data/synth_pmd"
CATEGORIES = ["A","B","C","D","E","F","G","H","I","J"]

# 카테고리 정의
# (shape, corner, fringe)
CAT_DEF = {
    "A": ("tilted_rect",      "sharp",   "vertical"),
    "B": ("tilted_rect",      "sharp",   "horizontal"),
    "C": ("tilted_rect",      "rounded", "vertical"),
    "D": ("tilted_rect",      "rounded", "horizontal"),
    "E": ("trapezoid",        "sharp",   "vertical"),
    "F": ("trapezoid",        "sharp",   "horizontal"),
    "G": ("trapezoid",        "rounded", "vertical"),
    "H": ("trapezoid",        "rounded", "horizontal"),
    "I": ("perspective_quad", "sharp",   "mixed"),
    "J": ("perspective_quad", "rounded", "mixed"),
}


# ── 1. 사각형 4점 생성 ────────────────────────────────────────────────────────

def _clip_pts(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """4점 좌표가 이미지 경계를 벗어나지 않도록 clip. margin=1px."""
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


def _is_inside(pts: np.ndarray, W: int, H: int) -> bool:
    """4점이 모두 이미지 내부에 있는지 확인."""
    return (pts[:, 0].min() >= 0 and pts[:, 0].max() <= W - 1 and
            pts[:, 1].min() >= 0 and pts[:, 1].max() <= H - 1)


def make_tilted_rect(W: int, H: int) -> np.ndarray:
    """기울어진 직사각형 4점 (픽셀). TL→TR→BR→BL 순서.
    회전 후 bounding box를 이미지 내부로 엄격히 제한."""
    margin = 2.0   # 경계 여유 픽셀

    for _ in range(200):
        area_ratio = random.uniform(0.30, 0.90)   # 30%~90%
        aspect     = random.uniform(0.5, 2.0)
        angle_deg  = random.uniform(-30, 30)

        rad   = math.radians(angle_deg)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))

        # 회전 후 bounding box 크기 공식:
        #   bbox_w = rw * cos_a + rh * sin_a
        #   bbox_h = rw * sin_a + rh * cos_a
        # bbox가 이미지를 넘지 않도록 rw/rh 최대값 계산
        max_bbox_w = W - margin * 2
        max_bbox_h = H - margin * 2

        # aspect = rw / rh 이므로 rh 기준으로 풀기
        #   rw = aspect * rh
        #   bbox_w = aspect*rh*cos_a + rh*sin_a = rh*(aspect*cos_a + sin_a)
        #   bbox_h = aspect*rh*sin_a + rh*cos_a = rh*(aspect*sin_a + cos_a)
        denom_w = aspect * cos_a + sin_a + 1e-8
        denom_h = aspect * sin_a + cos_a + 1e-8
        max_rh  = min(max_bbox_w / denom_w, max_bbox_h / denom_h)

        # 면적 기반 rh
        rect_area = W * H * area_ratio
        rh = math.sqrt(rect_area / aspect)
        rh = min(rh, max_rh)
        rw = aspect * rh

        # 실제 회전된 4점 계산
        hw, hh = rw / 2, rh / 2
        corners = np.array([
            [-hw, -hh], [ hw, -hh],
            [ hw,  hh], [-hw,  hh],
        ])
        rot     = np.array([[math.cos(rad), -math.sin(rad)],
                            [math.sin(rad),  math.cos(rad)]])
        rotated = (rot @ corners.T).T                  # (4, 2) 중심 기준

        # 회전된 bounding box
        bbox_w = rotated[:, 0].max() - rotated[:, 0].min()
        bbox_h = rotated[:, 1].max() - rotated[:, 1].min()

        # 중심점: bounding box가 이미지 내부에 완전히 들어오는 범위
        cx_min = bbox_w / 2 + margin
        cx_max = W - bbox_w / 2 - margin
        cy_min = bbox_h / 2 + margin
        cy_max = H - bbox_h / 2 - margin

        if cx_min > cx_max or cy_min > cy_max:
            continue   # 크기가 너무 커서 이미지 내부 배치 불가 → 재시도

        cx  = random.uniform(cx_min, cx_max)
        cy  = random.uniform(cy_min, cy_max)
        pts = (rotated + np.array([cx, cy])).astype(np.float32)

        if _is_inside(pts, W, H):
            return pts

    # fallback
    m = W * 0.1
    return np.array([[m,m],[W-m,m],[W-m,H-m],[m,H-m]], dtype=np.float32)


def make_trapezoid(W: int, H: int) -> np.ndarray:
    """카메라 앙각으로 인한 사다리꼴 4점."""
    margin = max(1.0, W * 0.02)

    for _ in range(100):
        ratio = random.uniform(0.5, 0.9)
        y_top = H * random.uniform(0.05, 0.35)   # 더 작은 사다리꼴 허용
        y_bot = H * random.uniform(0.65, 0.95)

        x_bl  = margin + random.uniform(0, W * 0.1)
        x_br  = W - margin - random.uniform(0, W * 0.1)
        w_bot = x_br - x_bl

        w_top  = w_bot * ratio
        offset = random.uniform(-w_bot * 0.05, w_bot * 0.05)
        cx     = (x_bl + x_br) / 2 + offset
        x_tl   = cx - w_top / 2
        x_tr   = cx + w_top / 2

        pts = np.array([
            [x_tl, y_top],
            [x_tr, y_top],
            [x_br, y_bot],
            [x_bl, y_bot],
        ], dtype=np.float32)

        if _is_inside(pts, W, H):
            return pts

    # fallback
    m = W * 0.1
    return np.array([[m,m],[W-m,m],[W-m,H-m],[m,H-m]], dtype=np.float32)


def make_perspective_quad(W: int, H: int) -> np.ndarray:
    """임의 볼록 사각형 4점. 이미지 경계 내부 보장."""
    margin = max(1.0, W * 0.02)

    for _ in range(100):
        tl = np.array([random.uniform(margin, W*0.30),
                       random.uniform(margin, H*0.30)])
        tr = np.array([random.uniform(W*0.70, W-margin),
                       random.uniform(margin, H*0.30)])
        br = np.array([random.uniform(W*0.70, W-margin),
                       random.uniform(H*0.70, H-margin)])
        bl = np.array([random.uniform(margin, W*0.30),
                       random.uniform(H*0.70, H-margin)])

        pts  = np.array([tl, tr, br, bl], dtype=np.float32)
        area = _shoelace_area(pts)

        if (0.30 <= area / (W * H) <= 0.90 and _is_inside(pts, W, H)):
            return pts

    m = W * 0.1
    return np.array([[m,m],[W-m,m],[W-m,H-m],[m,H-m]], dtype=np.float32)


def _shoelace_area(pts: np.ndarray) -> float:
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


# ── 2. 마스크 생성 (Sharp / Rounded) ─────────────────────────────────────────

def make_mask(pts: np.ndarray, W: int, H: int,
              corner: str, radius: int = 0) -> Image.Image:
    """
    4점 좌표로 사각형 마스크 생성.
    corner='rounded' 시 모서리 rounding 처리.
    """
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    poly = [(float(p[0]), float(p[1])) for p in pts]

    if corner == "sharp":
        draw.polygon(poly, fill=255)

    else:  # rounded
        r = radius if radius > 0 else random.randint(8, 25)
        _draw_rounded_polygon(draw, poly, r)

    return mask


def _draw_rounded_polygon(draw: ImageDraw.Draw,
                           pts: list[tuple], r: int) -> None:
    """
    각 꼭짓점을 내접 arc로 rounding 처리.
    직선 구간 + arc 샘플링 점만으로 polygon을 구성.
    별도의 원(ellipse)을 사용하지 않음.

    각 꼭짓점 처리:
      1. 꼭짓점 직전 r만큼 앞 → p_arc_start (직선 끝점)
      2. 꼭짓점 직후 r만큼 뒤 → p_arc_end   (직선 시작점)
      3. p_arc_start → p_arc_end 를 내접원 arc로 샘플링
    """
    n = len(pts)
    r = max(1, r)

    # 각 꼭짓점별 (p_arc_start, p_arc_end, center, ang_start, ang_end) 계산
    corners = []
    for i in range(n):
        p_prev = np.array(pts[(i - 1) % n], dtype=np.float64)
        p_curr = np.array(pts[i],           dtype=np.float64)
        p_next = np.array(pts[(i + 1) % n], dtype=np.float64)

        d_in  = p_curr - p_prev
        d_out = p_next - p_curr
        len_in  = np.linalg.norm(d_in)  + 1e-8
        len_out = np.linalg.norm(d_out) + 1e-8
        u_in  = d_in  / len_in
        u_out = d_out / len_out

        # r이 변 길이의 45% 초과하지 않도록 제한
        t = min(r, len_in  * 0.45)
        s = min(r, len_out * 0.45)
        rc = min(t, s)

        p_start = p_curr - u_in  * rc
        p_end   = p_curr + u_out * rc

        # 내접원 중심: 꼭짓점에서 내부 방향(이등분선)으로 이동
        bisect  = -(u_in + u_out)
        b_len   = np.linalg.norm(bisect) + 1e-8
        bisect  = bisect / b_len

        cos_a   = np.clip(np.dot(u_in, -u_out), -1.0, 1.0)
        half_a  = math.acos(cos_a) / 2.0
        sin_h   = math.sin(half_a)
        if sin_h < 1e-6:
            # 180도에 가까운 직선 꼭짓점 → arc 불필요
            corners.append((p_start, p_end, None, 0, 0, 0))
            continue

        dist_to_center = rc / (sin_h + 1e-8)
        center = p_curr + bisect * dist_to_center
        arc_r  = abs(dist_to_center * sin_h)

        ang_start = math.atan2(p_start[1] - center[1],
                               p_start[0] - center[0])
        ang_end   = math.atan2(p_end[1]   - center[1],
                               p_end[0]   - center[0])

        corners.append((p_start, p_end, center, ang_start, ang_end, arc_r))

    # 외곽선 좌표 수집: 직선 + arc 샘플링만 사용
    outline = []
    for i, (p_start, p_end, center, ang_start, ang_end, arc_r) in enumerate(corners):

        # 직선 구간: 이전 꼭짓점의 p_end → 현재 꼭짓점의 p_start
        outline.append(tuple(p_start))

        # arc 구간 샘플링 (center가 None이면 직선으로 대체)
        if center is None:
            outline.append(tuple(p_end))
            continue

        # 시계방향으로 각도 보정
        diff = ang_end - ang_start
        if diff < 0:
            diff += 2 * math.pi
        # 반시계 방향이면 반대로
        if diff > math.pi:
            diff -= 2 * math.pi

        n_samples = max(4, int(abs(math.degrees(diff)) / 5))
        for k in range(n_samples + 1):
            a  = ang_start + diff * k / n_samples
            px = center[0] + arc_r * math.cos(a)
            py = center[1] + arc_r * math.sin(a)
            outline.append((float(px), float(py)))

    if len(outline) >= 3:
        draw.polygon(outline, fill=255)


# ── 3. Fringe 패턴 생성 ───────────────────────────────────────────────────────

def make_fringe(W: int, H: int, direction: str) -> np.ndarray:
    """
    fringe 패턴 생성. float32 [0, 255].
    direction: 'vertical' | 'horizontal' | 'mixed'
    """
    A   = random.uniform(80,  180)
    B   = random.uniform(40,  100)
    phi = random.uniform(0,   2 * math.pi)
    # 줄무늬 폭 10~60px → 주파수 계산
    stripe_w = random.uniform(10, 60)
    f        = 1.0 / stripe_w

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)

    if direction == "vertical":
        fringe = A + B * np.cos(2 * math.pi * f * xs + phi)
        fringe = np.tile(fringe[np.newaxis, :], (H, 1))

    elif direction == "horizontal":
        phi2   = random.uniform(0, 2 * math.pi)
        fringe = A + B * np.cos(2 * math.pi * f * ys + phi2)
        fringe = np.tile(fringe[:, np.newaxis], (1, W))

    else:  # mixed
        phi2    = random.uniform(0, 2 * math.pi)
        f2      = 1.0 / random.uniform(10, 60)
        fv      = A/2 + B/2 * np.cos(2 * math.pi * f  * xs + phi)
        fh      = A/2 + B/2 * np.cos(2 * math.pi * f2 * ys + phi2)
        fringe  = fv[np.newaxis, :] + fh[:, np.newaxis]

    return np.clip(fringe, 0, 255).astype(np.float32)


# ── 4. 배경 생성 ──────────────────────────────────────────────────────────────

def make_background(W: int, H: int) -> np.ndarray:
    """단색 / 그라디언트 / 텍스처 중 랜덤 선택."""
    bg_type = random.choice(["flat", "gradient", "texture"])

    if bg_type == "flat":
        val = random.uniform(20, 90)
        bg  = np.full((H, W), val, dtype=np.float32)

    elif bg_type == "gradient":
        angle = random.uniform(0, math.pi)
        xs    = np.linspace(0, 1, W)
        ys    = np.linspace(0, 1, H)
        gx, gy = np.meshgrid(xs, ys)
        grad  = math.cos(angle) * gx + math.sin(angle) * gy
        v_min = random.uniform(15, 50)
        v_max = random.uniform(70, 110)
        bg    = (v_min + (v_max - v_min) * grad).astype(np.float32)

    else:  # texture (Perlin-like: sum of random sinusoids)
        bg = np.zeros((H, W), dtype=np.float32)
        for _ in range(4):
            fx  = random.uniform(0.005, 0.03)
            fy  = random.uniform(0.005, 0.03)
            phi = random.uniform(0, 2*math.pi)
            xs  = np.arange(W, dtype=np.float32)
            ys  = np.arange(H, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            bg += 12 * np.sin(2*math.pi*(fx*gx + fy*gy) + phi)
        bg = np.clip(bg + random.uniform(30, 70), 0, 255).astype(np.float32)

    return bg


# ── 5. 후처리 (노이즈 / 블러 / 조명) ─────────────────────────────────────────

def apply_postprocess(img_np: np.ndarray) -> np.ndarray:
    """노이즈, 블러, vignette 랜덤 적용."""
    # Gaussian noise
    if random.random() < 0.6:
        sigma    = random.uniform(3, 15)
        img_np  += np.random.normal(0, sigma, img_np.shape).astype(np.float32)

    # Salt & pepper
    if random.random() < 0.2:
        prob = random.uniform(0.001, 0.005)
        mask = np.random.random(img_np.shape)
        img_np[mask < prob]     = 0
        img_np[mask > 1 - prob] = 255

    img_np = np.clip(img_np, 0, 255)

    # Gaussian blur
    pil = Image.fromarray(img_np.astype(np.uint8))
    if random.random() < 0.3:
        sigma = random.uniform(0.5, 2.0)
        pil   = pil.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Motion blur (horizontal or vertical)
    if random.random() < 0.2:
        k = random.choice([3, 5])
        if random.random() < 0.5:
            # 수평 motion blur
            kernel = [0] * (k * k)
            for i in range(k):
                kernel[k // 2 * k + i] = 1
        else:
            # 수직 motion blur
            kernel = [0] * (k * k)
            for i in range(k):
                kernel[i * k + k // 2] = 1
        pil = pil.filter(ImageFilter.Kernel(
            size=(k, k),
            kernel=kernel,
            scale=k,
        ))

    # Vignette
    img_np = np.array(pil, dtype=np.float32)
    if random.random() < 0.4:
        H, W  = img_np.shape
        cx, cy = W/2, H/2
        xs    = np.linspace(-1, 1, W)
        ys    = np.linspace(-1, 1, H)
        gx, gy = np.meshgrid(xs, ys)
        dist  = np.sqrt(gx**2 + gy**2)
        vig   = 1.0 - random.uniform(0.2, 0.5) * dist
        img_np *= np.clip(vig, 0.3, 1.0)

    return np.clip(img_np, 0, 255).astype(np.uint8)


# ── 6. 단일 이미지 생성 ───────────────────────────────────────────────────────

def generate_one(cat: str, W: int = IMG_SIZE, H: int = IMG_SIZE,
                 max_retry: int = 200) -> tuple:
    """
    카테고리 하나에 대해 이미지 1장 생성.
    4점이 이미지 경계를 벗어나면 경계 내부가 될 때까지 재생성.

    Returns:
        img  : PIL Image (grayscale)
        pts  : np.ndarray (4, 2) 픽셀 좌표 [TL,TR,BR,BL]
        quad : list[float] normalized [0,1] 8개 좌표
    """
    shape, corner, fringe_dir = CAT_DEF[cat]

    for attempt in range(max_retry):

        # 1. 4점 생성
        if shape == "tilted_rect":
            pts = make_tilted_rect(W, H)
        elif shape == "trapezoid":
            pts = make_trapezoid(W, H)
        else:
            pts = make_perspective_quad(W, H)

        # 경계 검사: 벗어나면 재생성
        if not _is_inside(pts, W, H):
            continue

        # 2. 배경
        bg = make_background(W, H)

        # 3. Fringe 패턴
        fringe = make_fringe(W, H, fringe_dir)

        # 4. 마스크
        mask_img = make_mask(pts, W, H, corner)
        mask_np  = np.array(mask_img, dtype=np.float32) / 255.0

        # 5. 합성: mask 영역에 fringe, 외부에 배경
        composite = fringe * mask_np + bg * (1 - mask_np)

        # 6. 후처리
        final = apply_postprocess(composite)

        # 7. 정규화 좌표
        quad = [
            pts[0][0]/W, pts[0][1]/H,   # TL
            pts[1][0]/W, pts[1][1]/H,   # TR
            pts[2][0]/W, pts[2][1]/H,   # BR
            pts[3][0]/W, pts[3][1]/H,   # BL
        ]
        quad = [round(v, 6) for v in quad]

        return Image.fromarray(final, mode="L"), pts, quad

    raise RuntimeError(
        f"[{cat}] {max_retry}회 시도 후에도 유효한 좌표 생성 실패"
    )


# ── 7. 배치 생성 메인 ─────────────────────────────────────────────────────────

def generate_dataset(n_per_cat: int = 10, out_dir: str = OUT_DIR,
                     img_size: int = IMG_SIZE) -> None:
    """
    카테고리 A~J 각 n_per_cat 장 생성 → CSV 저장.

    Args:
        n_per_cat : 카테고리당 생성 수
        out_dir   : 저장 루트 폴더
        img_size  : 이미지 해상도
    """
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "metadata.csv")
    rows     = []
    idx      = 1

    for cat in CATEGORIES:
        print(f"  카테고리 {cat} 생성 중 ({n_per_cat}장)...", end=" ")
        count = 0

        for _ in range(n_per_cat):
            try:
                img, pts, quad = generate_one(cat, img_size, img_size)
            except RuntimeError as e:
                print(f"\n  [SKIP] {e}")
                continue

            fname    = f"{idx:06d}_{cat}.png"
            img_path = os.path.join(img_dir, fname)
            img.save(img_path)

            rel_path = os.path.join("images", fname)
            rows.append([rel_path] + quad + [cat])
            idx   += 1
            count += 1

        print(f"완료 ({count}장)")

    # CSV 저장
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path",
                         "x1","y1","x2","y2",
                         "x3","y3","x4","y4",
                         "category"])
        writer.writerows(rows)

    total = len(rows)
    print(f"\n총 {total}장 생성 완료 → {csv_path}")


# ── 8. 시각화 (확인용) ────────────────────────────────────────────────────────

def visualize_samples(out_dir: str = OUT_DIR, n_cols: int = 5) -> None:
    """생성된 이미지 샘플을 그리드로 시각화."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib 없음 — 시각화 스킵")
        return

    csv_path = os.path.join(out_dir, "metadata.csv")
    rows     = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 카테고리별 첫 번째 이미지만
    cat_rows = {}
    for row in rows:
        c = row["category"]
        if c not in cat_rows:
            cat_rows[c] = row

    n     = len(CATEGORIES)
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for i, cat in enumerate(CATEGORIES):
        ax  = axes[i]
        row = cat_rows.get(cat)
        if row is None:
            ax.axis("off")
            continue

        img_path = os.path.join(out_dir, row["image_path"])
        img      = Image.open(img_path)
        ax.imshow(img, cmap="gray")

        # 4점 표시
        coords = [float(row[k]) for k in
                  ["x1","y1","x2","y2","x3","y3","x4","y4"]]
        W, H   = img.size
        xs = [coords[i]*W for i in range(0, 8, 2)] + [coords[0]*W]
        ys = [coords[i]*H for i in range(1, 8, 2)] + [coords[1]*H]
        ax.plot(xs, ys, "r-", linewidth=1.5)
        ax.scatter(xs[:-1], ys[:-1], c="r", s=20, zorder=5)

        shape, corner, fringe = CAT_DEF[cat]
        ax.set_title(f"{cat}: {shape[:5]}/{corner[:3]}/{fringe[:3]}",
                     fontsize=8)
        ax.axis("off")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Synthetic PMD Fringe Samples (per Category)", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "samples.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"Visualization saved -> {save_path}")
    plt.show()


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=10,
                        help="카테고리당 생성 수 (기본 10)")
    parser.add_argument("--out_dir",  type=str, default=OUT_DIR)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--no_vis",   action="store_true",
                        help="시각화 스킵")
    args = parser.parse_args()

    print(f"=== PMD Fringe 합성 이미지 생성 ===")
    print(f"카테고리 : {CATEGORIES}")
    print(f"카테고리당 : {args.n}장  |  총 목표 : {args.n * len(CATEGORIES)}장")
    print(f"출력 경로 : {args.out_dir}")
    print()

    generate_dataset(
        n_per_cat = args.n,
        out_dir   = args.out_dir,
        img_size  = args.img_size,
    )

    if not args.no_vis:
        visualize_samples(args.out_dir)
