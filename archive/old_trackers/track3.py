import argparse
import cv2
import numpy as np

def parse_bbox(s: str):
    x, y, w, h = map(float, s.split(","))
    return (int(x), int(y), int(w), int(h))

def clamp_bbox(b, W, H):
    x, y, w, h = b
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)

def crop(img, bbox):
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def gaussian_2d(shape, sigma=2.0):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma * sigma))
    return g

class MOSSE:
    """
    Very fast correlation filter tracker (classical).
    Tracks translation within a fixed-size window.
    """
    def __init__(self, learning_rate=0.125, eps=1e-5):
        self.lr = learning_rate
        self.eps = eps
        self.A = None
        self.B = None
        self.G = None
        self.win = None
        self.size = None

    def _preprocess(self, x):
        x = x.astype(np.float32)
        x = np.log(x + 1.0)
        x = (x - x.mean()) / (x.std() + 1e-5)
        return x * self.win

    def init(self, gray, bbox):
        patch = crop(gray, bbox)
        h, w = patch.shape[:2]
        self.size = (w, h)

        # Hann window
        self.win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        # Desired response (Gaussian peak)
        g = gaussian_2d((h, w), sigma=max(2.0, min(h, w) / 10.0)).astype(np.float32)
        self.G = np.fft.fft2(g)

        x = self._preprocess(patch)
        F = np.fft.fft2(x)
        self.A = self.G * np.conj(F)
        self.B = F * np.conj(F)

    def update(self, gray, bbox):
        # bbox defines the search window (same size as init)
        patch = crop(gray, bbox)
        patch = cv2.resize(patch, self.size, interpolation=cv2.INTER_LINEAR)

        x = self._preprocess(patch)
        F = np.fft.fft2(x)

        H = self.A / (self.B + self.eps)
        resp = np.fft.ifft2(H * F).real

        # peak response location
        dy, dx = np.unravel_index(np.argmax(resp), resp.shape)

        # Confidence: peak-to-mean ratio (cheap proxy)
        peak = resp[dy, dx]
        mean = resp.mean()
        std = resp.std() + 1e-6
        psr_like = (peak - mean) / std

        # shift relative to center
        h, w = resp.shape
        cx, cy = w // 2, h // 2
        shift_x = dx - cx
        shift_y = dy - cy

        # Online update (classical)
        self.A = (1 - self.lr) * self.A + self.lr * (self.G * np.conj(F))
        self.B = (1 - self.lr) * self.B + self.lr * (F * np.conj(F))

        return shift_x, shift_y, float(psr_like)

def good_features(gray, bbox, max_corners=200):
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]
    pts = cv2.goodFeaturesToTrack(
        roi, maxCorners=max_corners, qualityLevel=0.01, minDistance=3, blockSize=7
    )
    if pts is None:
        return None
    pts[:, 0, 0] += x
    pts[:, 0, 1] += y
    return pts

def bbox_from_center(cx, cy, w, h):
    return (int(cx - w/2), int(cy - h/2), int(w), int(h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", required=True, help="x,y,w,h in first frame")
    ap.add_argument("--out", default="tracked.mp4")
    ap.add_argument("--maxdim", type=int, default=0, help="optional resize max dimension (0=off)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read video")

    # Optional resize for speed
    def maybe_resize(frame):
        if args.maxdim and max(frame.shape[:2]) > args.maxdim:
            H, W = frame.shape[:2]
            s = args.maxdim / float(max(H, W))
            frame = cv2.resize(frame, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        return frame

    frame0 = maybe_resize(frame0)
    H, W = frame0.shape[:2]
    bbox = clamp_bbox(parse_bbox(args.bbox), W, H)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # init trackers
    mosse = MOSSE(learning_rate=0.08)
    mosse.init(prev_gray, bbox)

    pts_prev = good_features(prev_gray, bbox)
    lost_count = 0

    def draw(frame, bbox, color=(0, 255, 0), txt=""):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if txt:
            cv2.putText(frame, txt, (x, max(0, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    draw(frame0, bbox, txt="init")
    vw.write(frame0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = maybe_resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        bbox = clamp_bbox(bbox, W, H)
        x, y, w, h = bbox
        cx, cy = x + w/2.0, y + h/2.0

        # --- 1) Optical flow predict ---
        flow_ok = False
        if pts_prev is not None and len(pts_prev) >= 10:
            pts_next, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pts_prev, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
            st = st.reshape(-1)
            p0 = pts_prev[st == 1].reshape(-1, 2)
            p1 = pts_next[st == 1].reshape(-1, 2)

            if len(p0) >= 10:
                M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3)
                if M is not None and inliers is not None:
                    inl = int(inliers.sum())
                    if inl >= 8:
                        # apply to center and size (scale from affine)
                        dx, dy = M[0, 2], M[1, 2]
                        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                        cx += dx
                        cy += dy
                        w = max(10, int(w * scale))
                        h = max(10, int(h * scale))
                        flow_ok = True

        pred_bbox = clamp_bbox(bbox_from_center(cx, cy, w, h), W, H)

        # --- 2) MOSSE refine in a search window ---
        # Search window: 2x bbox around predicted center
        sx = int(max(0, cx - (w)))
        sy = int(max(0, cy - (h)))
        sw = int(min(W - sx, 2*w))
        sh = int(min(H - sy, 2*h))
        search = (sx, sy, sw, sh)
        search = clamp_bbox(search, W, H)

        # MOSSE expects a window same size as init. Use predicted bbox-sized crop centered in search:
        # we’ll create a "mosse window bbox" centered at predicted center with mosse.size
        mw, mh = mosse.size
        mosse_win = clamp_bbox(bbox_from_center(cx, cy, mw, mh), W, H)

        shift_x, shift_y, conf = mosse.update(gray, mosse_win)
        cx += shift_x
        cy += shift_y
        refined = clamp_bbox(bbox_from_center(cx, cy, w, h), W, H)

        # --- 3) Confidence / recovery ---
        if conf < 5.0:  # tune per video
            lost_count += 1
        else:
            lost_count = 0

        if lost_count >= 10:
            # Cheap recovery: expand search using template matching on downscaled image
            tmpl = crop(prev_gray, clamp_bbox(bbox_from_center(cx, cy, mw, mh), W, H))
            if tmpl.size > 0:
                scale = 0.5
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                tmpl_s = cv2.resize(tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(small, tmpl_s, cv2.TM_CCOEFF_NORMED)
                _, vmax, _, vloc = cv2.minMaxLoc(res)
                if vmax > 0.4:
                    cx = (vloc[0] + tmpl_s.shape[1]/2) / scale
                    cy = (vloc[1] + tmpl_s.shape[0]/2) / scale
                    refined = clamp_bbox(bbox_from_center(cx, cy, w, h), W, H)
                    lost_count = 0

        bbox = refined

        # refresh features
        pts_prev = good_features(gray, bbox)
        prev_gray = gray

        draw(frame, bbox, txt=f"conf={conf:.1f}{' LOST' if lost_count else ''}")
        vw.write(frame)

    cap.release()
    vw.release()
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
