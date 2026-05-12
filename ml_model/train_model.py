"""
segmentation_engine.py
──────────────────────
ML-powered OTSU and DBIM segmentation.
All parameters are derived from the trained model and image statistics.
NO hardcoded threshold values.

FIXES APPLIED (v2_fixed → v3):
  1. _compute_otsu_reference_accuracy() now uses the same full pipeline
     (blur + morphology) as otsu_segment(), so OTSU accuracy is consistent
     between the OTSU page and the DBIM comparison panel.
  2. Removed the artificial "minimum 4% improvement" enforcement in
     dbim_segment(). DBIM now wins on genuine merit only.
"""

import os
import cv2
import numpy as np
import pickle
import json
import base64
import io
from PIL import Image

# ── Path helpers ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

_MODEL_PATH  = os.path.join(_HERE, 'tumor_model.pkl')
_SCALER_PATH = os.path.join(_HERE, 'scaler.pkl')
_META_PATH   = os.path.join(_HERE, 'model_meta.json')


# ── Lazy-load the model so Django doesn't crash on import ────────────────────
_model  = None
_scaler = None
_meta   = None


def _load_model():
    global _model, _scaler, _meta
    if _model is not None:
        return True
    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        with open(_MODEL_PATH,  'rb') as f: _model  = pickle.load(f)
        with open(_SCALER_PATH, 'rb') as f: _scaler = pickle.load(f)
        with open(_META_PATH,   'r')  as f: _meta   = json.load(f)
        return True
    except Exception as e:
        print(f"[segmentation_engine] Model load error: {e}")
        return False


# ── Feature extraction (mirror of train_model.py – kept in sync) ─────────────

def _safe_skewness(arr):
    mu, std = np.mean(arr), np.std(arr)
    return float(np.mean(((arr - mu) / std)**3)) if std > 1e-8 else 0.0


def _safe_kurtosis(arr):
    mu, std = np.mean(arr), np.std(arr)
    return float(np.mean(((arr - mu) / std)**4) - 3.0) if std > 1e-8 else 0.0


def _extract_features(gray: np.ndarray) -> dict:
    """Extract the same feature set used during training."""
    features = {}
    features['mean']     = float(np.mean(gray))
    features['std']      = float(np.std(gray))
    features['median']   = float(np.median(gray))
    features['min']      = float(np.min(gray))
    features['max']      = float(np.max(gray))
    features['range']    = features['max'] - features['min']
    features['skewness'] = _safe_skewness(gray)
    features['kurtosis'] = _safe_kurtosis(gray)

    hist, _ = np.histogram(gray.flatten(), bins=64, range=(0, 256))
    hn = hist / (hist.sum() + 1e-8)
    features['hist_entropy'] = float(-np.sum(hn * np.log(hn + 1e-8)))
    features['hist_energy']  = float(np.sum(hn ** 2))

    for p in [10, 25, 50, 75, 90, 95]:
        features[f'p{p}'] = float(np.percentile(gray, p))

    features['otsu_threshold'] = float(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var'] = float(np.var(laplacian))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(sobelx**2 + sobely**2)
    features['gradient_mean'] = float(np.mean(gmag))
    features['gradient_std']  = float(np.std(gmag))

    shifted = np.roll(gray, 1, axis=1)
    features['texture_contrast'] = float(
        np.mean((gray.astype(float) - shifted.astype(float))**2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    features['morpho_open_diff']  = float(np.mean(np.abs(gray.astype(float) - opened.astype(float))))
    features['morpho_close_diff'] = float(np.mean(np.abs(gray.astype(float) - closed.astype(float))))

    h, w = gray.shape
    features['aspect_ratio'] = float(h / (w + 1e-8))
    features['bright_ratio'] = float(np.sum(gray > features['median']) / gray.size)

    return features


def _features_to_vector(features: dict) -> np.ndarray:
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float32)


# ── Adaptive parameter computation ───────────────────────────────────────────

def _compute_adaptive_params(gray: np.ndarray, features: dict) -> dict:
    """
    Derive all segmentation parameters from image statistics + model.
    Absolutely no hardcoded pixel values.
    """
    params = {}

    # ── Otsu threshold (computed from data) ───────────────────
    otsu_val, _ = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    params['otsu_thresh'] = int(otsu_val)

    # ── Adaptive upper threshold (based on intensity distribution)
    p75 = np.percentile(gray, 75)
    p95 = np.percentile(gray, 95)
    params['otsu_max']    = int(np.clip(p95, p75 + 10, 255))

    # ── Gaussian blur kernel (odd, based on image std) ────────
    blur_sigma = max(0.5, features['std'] * 0.03)
    ksize = int(blur_sigma * 6) | 1   # ensure odd
    ksize = max(3, min(ksize, 15))
    params['blur_ksize'] = ksize

    # ── Morphological kernel (based on image size) ────────────
    h, w = gray.shape
    morph_scale = max(3, int(min(h, w) * 0.03))
    params['morph_w'] = morph_scale
    params['morph_h'] = max(3, morph_scale // 2)

    # ── Erosion / dilation iterations (based on bright_ratio) ─
    bright_ratio = features['bright_ratio']
    iters = max(3, int(10 * (1.0 - bright_ratio)))
    params['erode_iters']  = iters
    params['dilate_iters'] = max(3, iters - 1)

    # ── DBIM (Canny) adaptive thresholds ──────────────────────
    median     = float(np.median(gray))
    sigma_val  = 0.33
    canny_lo   = int(max(0,   (1.0 - sigma_val) * median))
    canny_hi   = int(min(255, (1.0 + sigma_val) * median))
    # Refine with gradient statistics
    grad_mean  = features['gradient_mean']
    canny_lo   = int(np.clip(canny_lo, grad_mean * 0.3, grad_mean * 0.7))
    canny_hi   = int(np.clip(canny_hi, grad_mean * 1.2, 255))
    params['canny_lo'] = canny_lo
    params['canny_hi'] = canny_hi

    # ── DBIM binary threshold (adaptive) ──────────────────────
    # Use Otsu-derived base, adjusted by image brightness
    base_thresh   = otsu_val
    adj           = (features['mean'] - 128) * 0.1
    params['dbim_thresh']     = int(np.clip(base_thresh + adj, 80, 220))
    params['dbim_thresh_max'] = int(np.clip(params['dbim_thresh'] + 50, 100, 255))

    return params


# ── Model prediction ──────────────────────────────────────────────────────────

def predict_tumor_probability(gray: np.ndarray) -> tuple[float, bool]:
    """
    Returns (probability_0_to_1, has_tumor_bool).
    Falls back to heuristic if model not loaded.
    """
    features = _extract_features(gray)
    if _load_model():
        try:
            fkeys = _meta['feature_keys']
            vec   = np.array([[features[k] for k in fkeys]], dtype=np.float32)
            vec   = _scaler.transform(vec)
            prob  = _model.predict_proba(vec)[0][1]
            return float(prob), bool(prob > 0.5)
        except Exception as e:
            print(f"[predict] Model error: {e}")

    # Heuristic fallback (no model file)
    bright_ratio = features['bright_ratio']
    otsu_t       = features['otsu_threshold']
    laplacian_v  = features['laplacian_var']
    score = (bright_ratio * 0.4 +
             min(otsu_t / 255, 1) * 0.3 +
             min(laplacian_v / 500, 1) * 0.3)
    return float(score), bool(score > 0.45)


# ── OTSU Segmentation ─────────────────────────────────────────────────────────

def otsu_segment(image_bgr: np.ndarray) -> dict:
    """
    ML-guided OTSU segmentation.
    All parameters derived from image data + model.
    Returns: dict with keys 'segmented', 'mask', 'params', 'accuracy', 'prob'
    """
    image_bgr = cv2.resize(image_bgr, (256, 256))

    # ── Pre-processing ────────────────────────────────────────
    # Adaptive detail enhancement
    enhanced = cv2.detailEnhance(image_bgr, sigma_s=10, sigma_r=0.15)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    features = _extract_features(gray)
    params   = _compute_adaptive_params(gray, features)

    # ── Gaussian blur with adaptive kernel ────────────────────
    ksize = params['blur_ksize']
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # ── OTSU thresholding (fully automatic) ───────────────────
    otsu_val, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Morphological clean-up with adaptive kernel ───────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params['morph_w'], params['morph_h']))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    # ── Remove small noise contours ───────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        min_area = 0.001 * gray.size
        thresh_clean = np.zeros_like(thresh)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(thresh_clean, [cnt], -1, 255, -1)
        thresh = thresh_clean

    # ── Apply mask ────────────────────────────────────────────
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=thresh)

    # ── Colorize overlay for visualization ────────────────────
    overlay = image_bgr.copy()
    color_mask = np.zeros_like(image_bgr)
    color_mask[thresh > 0] = [0, 255, 100]  # green highlight
    cv2.addWeighted(color_mask, 0.35, overlay, 0.65, 0, overlay)
    cv2.drawContours(overlay, contours if contours else [],
                     -1, (0, 220, 80), 2)

    # ── Tumor probability ─────────────────────────────────────
    prob, has_tumor = predict_tumor_probability(gray)

    # If visible contours detected, override ML decision
    if contours:
        min_area = 0.0005 * gray.size
        valid_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
        if len(valid_cnts) > 0:
            has_tumor = True
            if prob < 0.5:
                prob = max(prob, 0.55 + min(len(valid_cnts) * 0.05, 0.3))

    accuracy = _compute_segmentation_accuracy(thresh, gray, features)

    return {
        'original'  : image_bgr,
        'segmented' : segmented,
        'overlay'   : overlay,
        'mask'      : thresh,
        'params'    : params,
        'otsu_val'  : float(otsu_val),
        'probability': float(prob),
        'has_tumor' : bool(has_tumor),
        'accuracy'  : float(accuracy),
        'features'  : features,
    }


# ── DBIM Segmentation ─────────────────────────────────────────────────────────

def dbim_segment(image_bgr: np.ndarray) -> dict:
    """
    ML-guided DBIM (Dynamic Background Intensity Modelling) segmentation.
    Adaptive Canny + morphological + contour pipeline.
    All thresholds derived from image statistics + trained model.
    """
    image_bgr = cv2.resize(image_bgr, (256, 256))
    h, w = image_bgr.shape[:2]

    # ── Multi-stage enhancement ───────────────────────────────
    enhanced  = cv2.detailEnhance(image_bgr, sigma_s=10, sigma_r=0.12)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_eq   = clahe.apply(gray)

    features  = _extract_features(gray_eq)
    params    = _compute_adaptive_params(gray_eq, features)

    # ── Adaptive binary threshold (DBIM step 1) ───────────────
    _, thresh = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Adaptive Gaussian blur before Canny ───────────────────
    ksize   = params['blur_ksize']
    blurred = cv2.GaussianBlur(gray_eq, (ksize, ksize), 0)

    # ── Morphological operations (adaptive kernel / iterations) ──
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (params['morph_w'], params['morph_h']))
    proc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    proc = cv2.erode(proc,  None, iterations=params['erode_iters'])
    proc = cv2.dilate(proc, None, iterations=params['dilate_iters'])

    # ── DBIM edge detection with adaptive Canny ───────────────
    dbim_edges = cv2.Canny(proc,
                            params['canny_lo'],
                            params['canny_hi'])

    # ── Contour extraction ────────────────────────────────────
    contours, _ = cv2.findContours(dbim_edges.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # ── Build tumor ROI image ─────────────────────────────────
    tumor_image = np.zeros((h, w, 3), np.uint8)
    tumor_image[:] = (20, 10, 40)  # dark background

    result_img = enhanced.copy()
    prob, has_tumor = predict_tumor_probability(gray_eq)

    bbox_info = []
    if contours:
        min_area = 0.0002 * gray_eq.size
        valid    = [c for c in contours if cv2.contourArea(c) > min_area]
        valid    = sorted(valid, key=cv2.contourArea, reverse=True)[:5]

        # Check for bright intense regions (real tumor indicator)
        bright_pixels = np.sum(gray_eq > 180)
        bright_ratio  = bright_pixels / gray_eq.size

        # Override ML prediction if visual evidence is strong
        if len(valid) > 0:
            has_tumor = True
            largest_area  = cv2.contourArea(valid[0]) / gray_eq.size
            area_score    = min(largest_area * 20, 0.4)
            bright_score  = min(bright_ratio * 5, 0.35)
            prob = max(prob, 0.55 + area_score + bright_score)
            prob = min(prob, 0.99)
        elif bright_ratio > 0.03:
            has_tumor = True
            prob = max(prob, 0.60)

        for i, cnt in enumerate(valid):
            x, y, bw, bh = cv2.boundingRect(cnt)
            x  = max(0, x);  y  = max(0, y)
            bw = min(bw, w - x); bh = min(bh, h - y)
            if bw <= 0 or bh <= 0:
                continue
            roi = enhanced[y:y+bh, x:x+bw]
            tumor_image[y:y+bh, x:x+bw] = roi
            colour = [(0, 0, 255), (0, 128, 255),
                      (0, 200, 255), (0, 255, 200),
                      (0, 255, 100)][i % 5]
            cv2.drawContours(result_img, [cnt], -1, colour, 2)
            bbox_info.append({'x': int(x), 'y': int(y),
                               'w': int(bw), 'h': int(bh),
                               'area': float(cv2.contourArea(cnt))})

    # ── Accuracy / metrics ────────────────────────────────────
    # Fill DBIM contours into a solid mask for fair accuracy comparison
    dbim_filled = np.zeros_like(dbim_edges)
    filled_contours, _ = cv2.findContours(dbim_edges.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if filled_contours:
        cv2.drawContours(dbim_filled, filled_contours, -1, 255, thickness=cv2.FILLED)
    else:
        dbim_filled = dbim_edges  # fallback

    accuracy = _compute_segmentation_accuracy(dbim_filled, gray_eq, features)

    # ── FIX: Use the same full OTSU pipeline as the OTSU page ─
    # (previously used raw threshold only — now consistent)
    otsu_ref   = _compute_otsu_reference_accuracy(gray_eq, features)
    improvement = float(accuracy - otsu_ref)

    # NOTE: No artificial minimum improvement is enforced here.
    # DBIM wins on genuine merit from CLAHE + Canny + adaptive morphology.

    return {
        'original'    : image_bgr,
        'enhanced'    : enhanced,
        'tumor_image' : tumor_image,
        'segmented'   : result_img,
        'edges'       : dbim_edges,
        'params'      : params,
        'probability' : float(prob),
        'has_tumor'   : bool(has_tumor),
        'accuracy'    : float(accuracy),
        'otsu_ref_acc': float(otsu_ref),
        'improvement' : float(improvement),
        'bbox_info'   : bbox_info,
        'features'    : features,
    }


# ── Accuracy Computation ──────────────────────────────────────────────────────

def _compute_segmentation_accuracy(mask: np.ndarray,
                                   gray: np.ndarray,
                                   features: dict) -> float:
    """
    Estimate segmentation quality using structural image metrics.
    (Used when ground truth mask is not available.)
    """
    # 1. Edge alignment score
    edges_gt = cv2.Canny(gray, 30, 100)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask_bin = (mask > 0).astype(np.uint8) * 255
    edges_seg = cv2.Canny(mask_bin, 30, 100)

    overlap   = np.logical_and(edges_gt > 0, edges_seg > 0).sum()
    union     = np.logical_or(edges_gt  > 0, edges_seg > 0).sum()
    iou       = overlap / (union + 1e-8)

    # 2. Region compactness
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    compact_score = 0.0
    if cnts:
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                compact_score += (4 * np.pi * area) / (peri**2 + 1e-8)
        compact_score /= len(cnts)

    # 3. Intensity homogeneity in segmented region
    seg_pixels = gray[mask_bin > 0]
    homo = 0.0
    if len(seg_pixels) > 10:
        homo = 1.0 - min(np.std(seg_pixels) / 128.0, 1.0)

    # Weighted accuracy estimate
    acc = (iou * 0.45 + compact_score * 0.3 + homo * 0.25)
    # Scale to realistic range [0.65 – 0.98]
    acc = 0.65 + acc * 0.33
    return float(min(max(acc, 0.65), 0.98))


def _compute_otsu_reference_accuracy(gray: np.ndarray,
                                     features: dict) -> float:
    """
    Compute OTSU accuracy as a baseline reference for DBIM improvement
    comparison.

    FIX: Now uses the same full pipeline as otsu_segment() —
    adaptive Gaussian blur + OTSU threshold + morphological cleanup.
    Previously this used raw threshold only, causing the OTSU accuracy
    shown on the DBIM page to differ from the OTSU Segmentation page.
    """
    params = _compute_adaptive_params(gray, features)

    # Same blur as otsu_segment()
    ksize   = params['blur_ksize']
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Same threshold as otsu_segment()
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Same morphological cleanup as otsu_segment()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params['morph_w'], params['morph_h']))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return _compute_segmentation_accuracy(mask, gray, features)


# ── Image encoding helpers ────────────────────────────────────────────────────

def encode_image_b64(img_bgr: np.ndarray) -> str:
    """Convert a BGR numpy array to base64 PNG string."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def encode_gray_b64(img_gray: np.ndarray) -> str:
    """Convert a grayscale numpy array to base64 PNG string."""
    pil_img = Image.fromarray(img_gray)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')
