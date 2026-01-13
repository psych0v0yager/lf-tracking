"""
Robust Single Object Tracker - Version 14

Simpler approach:
- NO motion-based reinitialization (this was causing problems)
- Just let CSRT run
- Verify each frame against template
- Use template search for recovery when verification fails

Usage:
    python best_tracker_v14.py test_track_dron1.mp4 1314 623 73 46
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import argparse


def create_tracker():
    return cv2.TrackerCSRT_create(), "CSRT"


def clamp_bbox(bbox, frame_shape):
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    x = max(0, min(int(x), fw - int(w) - 1))
    y = max(0, min(int(y), fh - int(h) - 1))
    return (x, y, int(w), int(h))


class TemplateVerifier:
    def __init__(self):
        self.reference_template = None
        self.template_size = None

    def set_reference(self, gray, bbox):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w > 0 and h > 0:
            self.reference_template = gray[y:y+h, x:x+w].copy()
            self.template_size = (w, h)

    def verify(self, gray, bbox, threshold=0.3):
        if self.reference_template is None:
            return True, 1.0

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w <= 0 or h <= 0:
            return False, 0.0

        current = gray[y:y+h, x:x+w]

        if current.size == 0:
            return False, 0.0

        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except Exception:
                return False, 0.0

        try:
            result = cv2.matchTemplate(current, self.reference_template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0] if result.size > 0 else 0.0
        except Exception:
            try:
                score = np.corrcoef(current.flatten(), self.reference_template.flatten())[0, 1]
                if np.isnan(score):
                    score = 0.0
            except Exception:
                score = 0.0

        return score >= threshold, score

    def search(self, gray, search_region, threshold=0.25):
        if self.reference_template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        th, tw = self.reference_template.shape[:2]

        if sw <= tw or sh <= th:
            return None

        region = gray[sy:sy+sh, sx:sx+sw]

        best_val = 0
        best_loc = None
        best_scale = 1.0

        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h, new_w = int(th * scale), int(tw * scale)
            if new_h >= region.shape[0] or new_w >= region.shape[1]:
                continue
            if new_h < 5 or new_w < 5:
                continue

            try:
                scaled = cv2.resize(self.reference_template, (new_w, new_h))
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
            except Exception:
                continue

        if best_val >= threshold and best_loc is not None:
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            return (x, y, w, h, best_val)

        return None


class RobustTrackerV14:
    def __init__(self, verify_threshold=0.25):
        self.verify_threshold = verify_threshold

        self.tracker = None
        self.tracker_name = None
        self.verifier = TemplateVerifier()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None
        self.verify_score = 1.0

        self.last_good_bbox = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.verifier.set_reference(gray, self.bbox)

        self.tracker, self.tracker_name = create_tracker()

        try:
            self.tracker.init(frame, self.bbox)
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        # Just run CSRT - no motion-based reinitialization
        try:
            tracker_success, box = self.tracker.update(frame)
        except Exception:
            tracker_success = False
            box = self.bbox

        # Verify the result
        actual_success = False
        if tracker_success:
            temp_bbox = tuple(int(v) for v in box)
            is_valid, self.verify_score = self.verifier.verify(gray, temp_bbox, self.verify_threshold)

            if is_valid:
                x, y, w, h = temp_bbox
                orig_w, orig_h = self.original_size

                w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
                h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
                cx, cy = x + box[2]/2, y + box[3]/2
                self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

                self.lost_count = 0
                self.last_good_bbox = self.bbox
                actual_success = True
        else:
            self.verify_score = 0.0

        # Recovery when tracking fails or verification fails
        if not actual_success:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            # Search around last good position
            if self.last_good_bbox is not None:
                lg_cx = self.last_good_bbox[0] + orig_w / 2
                lg_cy = self.last_good_bbox[1] + orig_h / 2

                expand = 3.0 + self.lost_count * 0.5
                search_w = orig_w * expand
                search_h = orig_h * expand

                search_region = (
                    int(max(0, lg_cx - search_w / 2)),
                    int(max(0, lg_cy - search_h / 2)),
                    int(min(search_w, fw)),
                    int(min(search_h, fh))
                )

                match = self.verifier.search(gray, search_region, threshold=0.2)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                    self.verify_score = score

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"local({score:.2f})"
                    self.last_good_bbox = self.bbox
                    actual_success = True

            # Wide search
            if not actual_success:
                match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.15)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                    self.verify_score = score

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"wide({score:.2f})"
                    self.last_good_bbox = self.bbox
                    actual_success = True

            # Hold position as last resort
            if not actual_success:
                if self.last_good_bbox is not None:
                    self.bbox = self.last_good_bbox
                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)
                self.recovery_method = "hold"

        return actual_success, self.bbox, self.verify_score


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V14")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--verify-threshold", type=float, default=0.25)
    parser.add_argument("--output", "-o", help="Output video path")

    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video: {args.video}")
    print(f"Initial bbox: {bbox}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    fh, fw = frame.shape[:2]

    tracker = RobustTrackerV14(verify_threshold=args.verify_threshold)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"V14: Pure CSRT + Verification (no motion reinit)")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v14.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        success, bbox, verify_score = tracker.update(frame)

        if success:
            success_count += 1

        x, y, w, h = bbox

        if tracker.recovery_method:
            color = (0, 255, 255)
        elif verify_score > 0.4:
            color = (0, 255, 0)
        elif verify_score > 0.25:
            color = (0, 200, 200)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Verify: {verify_score:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V14", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{frame_num} frames ({100*success_count/frame_num:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
