"""
SyncNature — Signature Video Analyzer

Takes a recorded signature video and processes it through 4 numerical methods:

  1. Edge Detection  — isolate stroke pixels, track pen tip per frame, Sobel edges
  2. Interpolation   — arc-length parameterized cubic spline through pen positions
  3. Derivatives     — velocity and acceleration via central difference
  4. Rootfinding     — bisection + Newton-Raphson finds where ds/dt = 0 (speed minima)
"""

from flask import Flask, request, jsonify, send_file
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import cv2
import tempfile
import os

app = Flask(__name__)


@app.route('/')
def index():
    return send_file('index.html')


# ══════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    frames = []
    times_ms = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        times_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        frames.append(frame)
    cap.release()

    if len(frames) < 4:
        return None, None

    # FPS from actual timestamps — WebM header is unreliable
    if len(times_ms) > 1 and times_ms[-1] > 0:
        total_ms = times_ms[-1] - times_ms[0]
        if total_ms > 0:
            fps = float(len(times_ms) - 1) / (total_ms / 1000.0)
            fps = float(np.clip(fps, 10.0, 60.0))
        else:
            fps = 30.0
    else:
        fps = 30.0

    return frames, fps


# ══════════════════════════════════════════════════════════════
# METHOD 1 — EDGE DETECTION
# ──────────────────────────────────────────────────────────────
# The canvas draws cyan strokes (#00f5ff) on black background.
# Cyan has high Blue, high Green, low Red — easy to isolate.
#
# Per frame:
#   mask[t] = (Blue > 80) AND (Green > 60) AND (Red < 120)
#   new_pixels[t] = mask[t] AND NOT mask[t-1]
#   tip[t] = centroid(new_pixels[t])
#
# This gives us one pen position per frame where new ink appears.
# Gaussian smoothing [1,2,4,2,1]/10 removes centroid jitter.
#
# Sobel edge detection on the final complete stroke image:
#   Gx = ∂I/∂x,  Gy = ∂I/∂y  (Sobel 3×3 kernel)
#   G  = √(Gx² + Gy²)
#   High G → rapid intensity change → stroke boundary
# ══════════════════════════════════════════════════════════════
def edge_detection(frames, fps):
    if len(frames) < 4:
        return None

    def get_mask(frame):
        b = frame[:, :, 0].astype(float)
        g = frame[:, :, 1].astype(float)
        r = frame[:, :, 2].astype(float)
        return (b > 80) & (g > 60) & (r < 120)

    # Track pen tip: centroid of new pixels each frame
    prev_mask     = get_mask(frames[0])
    pen_positions = []

    for i in range(1, len(frames)):
        curr_mask  = get_mask(frames[i])
        new_pixels = curr_mask & ~prev_mask
        n_new      = int(np.sum(new_pixels))

        if n_new >= 3:
            ys, xs = np.where(new_pixels)
            pen_positions.append({
                'x': float(np.mean(xs)),
                'y': float(np.mean(ys)),
                't': float(i) / fps * 1000.0  # ms
            })

        prev_mask = curr_mask

    if len(pen_positions) < 6:
        return None

    # Gaussian smoothing: [1,2,4,2,1]/10
    if len(pen_positions) >= 7:
        xs = np.array([p['x'] for p in pen_positions])
        ys = np.array([p['y'] for p in pen_positions])
        ts = [p['t'] for p in pen_positions]
        w  = np.array([1.0, 2.0, 4.0, 2.0, 1.0]); w /= w.sum()
        xs_s = np.convolve(xs, w, mode='same')
        ys_s = np.convolve(ys, w, mode='same')
        for j in range(3):
            xs_s[j]    = xs[j];    ys_s[j]    = ys[j]
            xs_s[-j-1] = xs[-j-1]; ys_s[-j-1] = ys[-j-1]
        pen_positions = [
            {'x': float(xs_s[j]), 'y': float(ys_s[j]), 't': float(ts[j])}
            for j in range(len(ts))
        ]

    # Sobel on final frame — geometric stroke boundaries
    final_mask = get_mask(frames[-1]).astype(np.uint8) * 255
    blurred    = cv2.GaussianBlur(final_mask, (3, 3), 0)
    sobelx     = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely     = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag  = np.sqrt(sobelx**2 + sobely**2)
    max_mag    = float(np.max(sobel_mag))
    thresh     = max_mag * 0.30 if max_mag > 0 else 1.0
    eys, exs   = np.where(sobel_mag > thresh)

    edge_points = []
    if len(exs) > 0:
        step = max(1, len(exs) // 300)
        for j in range(0, len(exs), step):
            edge_points.append([float(exs[j]), float(eys[j])])

    return {
        'pen_positions': pen_positions,
        'edge_points':   edge_points,
        'n_frames':      int(len(frames)),
        'fps':           float(fps)
    }


# ══════════════════════════════════════════════════════════════
# METHOD 2 — INTERPOLATION
# ──────────────────────────────────────────────────────────────
# The pen positions are discrete samples at ~15fps.
# Between frames the pen moved continuously — interpolation
# reconstructs that smooth continuous path.
#
# Arc-length parameterization:
#   s[0] = 0
#   s[i] = s[i-1] + ||pos[i] - pos[i-1]||
#   Makes parameter increase with distance, not time.
#   Uniform s sampling → uniform spatial sampling regardless of speed.
#
# CubicSpline:
#   Piecewise cubic, C2-continuous at every knot.
#   Passes exactly through all input points (residual = 0).
#   Between knots s_i and s_{i+1}:
#     x(s) = a + b(s-s_i) + c(s-s_i)² + d(s-s_i)³
# ══════════════════════════════════════════════════════════════
def interpolation(pen_positions):
    if len(pen_positions) < 4:
        return None

    pts = np.array([[p['x'], p['y']] for p in pen_positions], dtype=float)

    # Pre-smooth with Gaussian [1,2,4,2,1]/10
    if len(pts) >= 7:
        w    = np.array([1.0, 2.0, 4.0, 2.0, 1.0]); w /= w.sum()
        xs_s = np.convolve(pts[:, 0], w, mode='same')
        ys_s = np.convolve(pts[:, 1], w, mode='same')
        for j in range(3):
            xs_s[j]    = pts[j, 0];    ys_s[j]    = pts[j, 1]
            xs_s[-j-1] = pts[-j-1, 0]; ys_s[-j-1] = pts[-j-1, 1]
        pts = np.stack([xs_s, ys_s], axis=1)

    # Arc-length parameterization
    diffs    = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc_s    = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(arc_s[-1])
    if total_len < 5.0:
        return None

    # Remove duplicates — CubicSpline needs strictly increasing x
    _, unique_idx = np.unique(arc_s, return_index=True)
    arc_s_u = arc_s[unique_idx]
    pts_u   = pts[unique_idx]
    if len(arc_s_u) < 4:
        return None

    # Fit cubic splines x(s) and y(s)
    cs_x = CubicSpline(arc_s_u, pts_u[:, 0])
    cs_y = CubicSpline(arc_s_u, pts_u[:, 1])

    # Evaluate at N uniform arc-length positions
    N         = min(400, max(100, len(pen_positions) * 4))
    s_uniform = np.linspace(0.0, total_len, N)
    curve     = [
        {'x': float(cs_x(s_uniform[i])), 'y': float(cs_y(s_uniform[i]))}
        for i in range(N)
    ]

    # Verify: spline passes through all knots (residual should be ~0)
    max_residual = max(
        np.sqrt((float(cs_x(s)) - pts_u[i, 0])**2 +
                (float(cs_y(s)) - pts_u[i, 1])**2)
        for i, s in enumerate(arc_s_u)
    )

    return {
        'curve':        curve,
        'total_len':    float(total_len),
        'n_points':     int(N),
        'max_residual': float(max_residual)
    }


# ══════════════════════════════════════════════════════════════
# METHOD 3 — DERIVATIVES
# ──────────────────────────────────────────────────────────────
# Pen positions have actual timestamps — non-uniform spacing
# because some frames have no new pixels and are skipped.
#
# numpy.gradient handles non-uniform spacing correctly:
#   Central difference (interior):
#     vx[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
#   Forward/backward (endpoints):
#     vx[0] = (x[1] - x[0]) / (t[1] - t[0])
#
# Velocity:     speed = √(vx² + vy²)         [px/s]
# Acceleration: accel = √(ax² + ay²)         [px/s²]
# ds/dt:        signed derivative of speed   [px/s²]
#               → feeds rootfinding
#
# Savitzky-Golay smoothing preserves peak shapes.
# ══════════════════════════════════════════════════════════════
def derivatives(pen_positions, fps):
    if len(pen_positions) < 4:
        return None

    pts   = np.array([[p['x'], p['y']] for p in pen_positions], dtype=float)
    times = np.array([p['t'] for p in pen_positions], dtype=float)
    t_sec = times / 1000.0

    # Ensure strictly increasing timestamps
    for i in range(1, len(t_sec)):
        if t_sec[i] <= t_sec[i - 1]:
            t_sec[i] = t_sec[i - 1] + (1.0 / fps)

    # Central difference — non-uniform timestamps handled correctly
    vx    = np.gradient(pts[:, 0], t_sec)
    vy    = np.gradient(pts[:, 1], t_sec)
    speed = np.sqrt(vx**2 + vy**2)

    ax    = np.gradient(vx, t_sec)
    ay    = np.gradient(vy, t_sec)
    accel = np.sqrt(ax**2 + ay**2)

    # ds/dt — signed rate of speed change, feeds rootfinding
    ds_dt = np.gradient(speed, t_sec)

    # Savitzky-Golay smoothing
    n         = len(speed)
    win       = min(15, n if n % 2 == 1 else n - 1)
    win       = max(win, 5)
    if win % 2 == 0: win += 1
    polyorder = min(3, win - 1)

    if n >= win:
        speed_s = savgol_filter(speed, win, polyorder)
        ds_dt_s = savgol_filter(ds_dt, win, polyorder)
        accel_s = savgol_filter(accel, win, polyorder)
    else:
        speed_s = speed.copy()
        ds_dt_s = ds_dt.copy()
        accel_s = accel.copy()

    speed_s = np.maximum(speed_s, 0.0)
    accel_s = np.maximum(accel_s, 0.0)

    max_speed = float(np.max(speed_s)) if np.max(speed_s) > 1e-8 else 1.0
    max_accel = float(np.max(accel_s)) if np.max(accel_s) > 1e-8 else 1.0

    return {
        'speed':       speed.tolist(),        # raw speed (for display)
        'norm_speed':  (speed_s / max_speed).tolist(),
        'norm_accel':  (accel_s / max_accel).tolist(),
        'accel':       accel.tolist(),
        'ds_dt':       (ds_dt_s / max_speed).tolist(),
        'max_speed':   float(max_speed),
        'max_accel':   float(max_accel),
    }


# ══════════════════════════════════════════════════════════════
# METHOD 4 — ROOTFINDING
# ──────────────────────────────────────────────────────────────
# Input:  ds_dt — signed rate of speed change (from derivatives)
# Output: indices where ds/dt = 0, i.e. speed is at a local minimum
#
# A speed minimum means the pen slowed to its lowest point —
# a natural pause, hesitation, or direction reversal. These
# are the meaningful events in a signature's dynamics.
#
# ds/dt = 0 at every speed minimum. To find them exactly:
#
# Phase A — Bisection (guaranteed convergence):
#   Scan for sign change where ds/dt goes negative → positive.
#   This identifies a minimum (not maximum).
#   For each such interval [i, i+1] where f(i)·f(i+1) < 0:
#     mid = (a+b)/2
#     if f(a)·f(mid) < 0: b = mid   (root in left half)
#     else:               a = mid   (root in right half)
#     repeat until |b-a| < EPS = 1e-6
#
# Phase B — Newton-Raphson refinement:
#   Starting from bisection result x:
#     x_{n+1} = x_n - f(x_n) / f'(x_n)
#   f(x)  = CubicSpline(ds_dt)(x)      — value of ds/dt at x
#   f'(x) = CubicSpline(ds_dt)(x, 1)  — analytic derivative
#   Until |x_{n+1} - x_n| < EPS
#
# Filters applied before bisection:
#   - Must be a minimum: ds/dt goes − → + (not + → −)
#   - Speed at that point must be below 20% of max (genuine slow)
#   - Dip must be prominent: >15% drop from surroundings (not noise)
# ══════════════════════════════════════════════════════════════
def rootfinding(norm_speed, ds_dt):
    speed = np.array(norm_speed, dtype=float)
    dspdt = np.array(ds_dt,      dtype=float)
    n     = len(speed)

    if n < 4:
        return {'zeros': [], 'zero_positions': [], 'count': 0}

    EPS          = 1e-6
    MAX_ITER     = 100
    SPEED_THRESH = 0.20   # speed must be below 20% of max
    MIN_PROM     = 0.15   # dip must be at least 15% below surroundings

    t_idx  = np.arange(n, dtype=float)
    cs_dsp = CubicSpline(t_idx, dspdt)

    zeros = []

    for i in range(n - 1):
        # Condition 1: sign change with tolerance
        if dspdt[i] * dspdt[i + 1] >= -1e-6:
            continue
        # Condition 2: minimum only — ds/dt goes negative → positive
        if dspdt[i] >= 0:
            continue
        # Condition 3: speed genuinely low at this point
        local_speed = float(np.min(speed[max(0, i-2):min(n, i+3)]))
        if local_speed > SPEED_THRESH:
            continue
        # Condition 4: prominent dip — not just noise
        win_lo = max(0, i - 15)
        win_hi = min(n, i + 15)
        if float(np.max(speed[win_lo:win_hi])) - local_speed < MIN_PROM:
            continue

        # Phase A: Bisection
        a = float(i)
        b = float(i + 1)
        for _ in range(MAX_ITER):
            mid = (a + b) / 2.0
            if float(cs_dsp(a)) * float(cs_dsp(mid)) < 0:
                b = mid
            else:
                a = mid
            if (b - a) < EPS:
                break
        x = (a + b) / 2.0

        # Phase B: Newton-Raphson
        for _ in range(MAX_ITER):
            f  = float(cs_dsp(x))
            fp = float(cs_dsp(x, 1))
            if abs(fp) < 1e-12:
                break
            x_new = float(np.clip(x - f / fp, 0.0, float(n - 1)))
            if abs(x_new - x) < EPS:
                x = x_new
                break
            x = x_new

        zeros.append(int(np.clip(round(x), 0, n - 1)))

    # Deduplicate — zeros within 10 positions are the same event
    deduped  = []
    last_idx = -15
    for z in sorted(zeros):
        if z - last_idx > 10:
            deduped.append(int(z))
            last_idx = z

    return {
        'zeros':          deduped,
        'zero_positions': [float(z) / max(n - 1, 1) for z in deduped],
        'count':          int(len(deduped))
    }


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(video_path):
    frames, fps = extract_frames(video_path)
    if frames is None:
        return None, 'Could not read video'

    # Method 1
    edge = edge_detection(frames, fps)
    if edge is None:
        return None, 'Edge detection failed — no cyan stroke found'

    pen = edge['pen_positions']
    if len(pen) < 6:
        return None, 'Too few positions — draw larger or more slowly'

    # Method 2
    interp = interpolation(pen)
    if interp is None:
        return None, 'Interpolation failed'

    # Method 3
    deriv = derivatives(pen, fps)
    if deriv is None:
        return None, 'Derivative computation failed'

    # Method 4 — rootfinding on ds/dt to find speed minima
    roots = rootfinding(deriv['norm_speed'], deriv['ds_dt'])

    return {
        'pen_positions': pen,
        'edge_points':   edge['edge_points'],
        'curve':         interp['curve'],
        'total_len':     float(interp['total_len']),
        'max_residual':  float(interp['max_residual']),
        'norm_speed':    deriv['norm_speed'],
        'norm_accel':    deriv['norm_accel'],
        'accel':         deriv['accel'],
        'ds_dt':         deriv['ds_dt'],
        'max_speed':     float(deriv['max_speed']),
        'zeros':         roots['zeros'],
        'zero_positions':roots['zero_positions'],
        'zero_count':    int(roots['count']),
        'n_frames':      int(len(frames)),
        'fps':           float(fps),
        'n_pen':         int(len(pen)),
    }, None


# ══════════════════════════════════════════════════════════════
# FLASK
# ══════════════════════════════════════════════════════════════
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    vf = request.files['video']
    suffix = '.webm' if (vf.filename or '').endswith('.webm') else '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        vf.save(tmp.name)
        tmp_path = tmp.name

    try:
        result, err = run_pipeline(tmp_path)
        if result is None:
            return jsonify({'error': err}), 400
        return jsonify(result)
    except Exception as e:
        import traceback
        print('ERROR:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass


if __name__ == '__main__':
    print("=" * 50)
    print("  SyncNature — Signature Video Analyzer")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, use_reloader=False)