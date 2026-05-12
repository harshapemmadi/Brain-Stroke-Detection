"""
views.py – NeuroScan Pro
All segmentation uses the ML-powered segmentation engine (no hardcoded values).
"""
import os
import sys
import cv2
import numpy as np
import hashlib
import json
import re

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

ML_DIR = os.path.join(settings.BASE_DIR, 'ml_model')
if ML_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(ML_DIR))

try:
    from segmentation_engine import (
        otsu_segment, dbim_segment,
        encode_image_b64, encode_gray_b64,
        predict_tumor_probability,
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"[views] Engine not available: {e}")

from .models import UserProfile, SegmentationResult


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _get_current_user(request):
    uid = request.session.get('user_id')
    if uid:
        try:
            return UserProfile.objects.get(pk=uid)
        except UserProfile.DoesNotExist:
            pass
    return None


VALID_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'live.com', 'icloud.com', 'protonmail.com', 'aol.com',
    'mail.com', 'zoho.com', 'yandex.com', 'rediffmail.com',
    'msn.com', 'me.com', 'pm.me', 'fastmail.com',
]

def _validate_email(email):
    if not email:
        return 'Email is required.'
    if '@' not in email:
        return 'Enter a valid email address.'
    parts = email.split('@')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return 'Enter a valid email address.'
    domain = parts[1].lower()
    if domain not in VALID_EMAIL_DOMAINS:
        return 'Email must use a recognised provider (e.g. @gmail.com, @yahoo.com).'
    return None


def _validate_password(pw):
    if len(pw) < 8:
        return 'Password must be at least 8 characters.'
    if not re.search(r'[A-Za-z]', pw):
        return 'Password must contain at least one letter.'
    if not re.search(r'[0-9]', pw):
        return 'Password must contain at least one number.'
    if not re.search(r'[^A-Za-z0-9]', pw):
        return 'Password must contain at least one special character.'
    return None


# ── Public pages ──────────────────────────────────────────────────────────────

def index(request):
    user = _get_current_user(request)
    return render(request, 'index.html', {'user': user})


def login_page(request):
    if _get_current_user(request):
        return redirect('dashboard')
    return render(request, 'login.html', {})


def register_page(request):
    if _get_current_user(request):
        return redirect('dashboard')
    return render(request, 'register.html', {})


def logout_view(request):
    request.session.flush()
    return redirect('index')


# ── AJAX check endpoints ──────────────────────────────────────────────────────

def check_username(request):
    username = request.GET.get('username', '').strip()
    if not username:
        return JsonResponse({'taken': False})
    taken = UserProfile.objects.filter(username=username).exists()
    return JsonResponse({'taken': taken})


def check_phone(request):
    phone = request.GET.get('phone', '').strip()
    if not phone:
        return JsonResponse({'taken': False})
    taken = UserProfile.objects.filter(contact=phone).exists()
    return JsonResponse({'taken': taken})


def check_email_ajax(request):
    email = request.GET.get('email', '').strip()
    if not email:
        return JsonResponse({'taken': False, 'invalid': False})
    err = _validate_email(email)
    if err:
        return JsonResponse({'taken': False, 'invalid': True, 'msg': err})
    taken = UserProfile.objects.filter(email=email).exists()
    return JsonResponse({'taken': taken, 'invalid': False})


# ── Auth actions ──────────────────────────────────────────────────────────────

def login_action(request):
    if request.method != 'POST':
        return redirect('login')

    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '').strip()

    if not username or not password:
        return render(request, 'login.html',
                      {'error': 'Please fill in all fields.'})

    pw_hash = _hash_password(password)
    try:
        user = UserProfile.objects.get(username=username, password=pw_hash)
        request.session['user_id'] = user.pk
        request.session['username'] = user.username
        return redirect('dashboard')
    except UserProfile.DoesNotExist:
        return render(request, 'login.html',
                      {'error': 'Invalid username or password.'})


def register_action(request):
    if request.method != 'POST':
        return redirect('register')

    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '').strip()
    confirm  = request.POST.get('confirm_password', '').strip()
    email    = request.POST.get('email', '').strip()
    contact  = request.POST.get('contact', '').strip()
    address  = request.POST.get('address', '').strip()

    errors = {}

    if not username:
        errors['username'] = 'Username is required.'
    elif UserProfile.objects.filter(username=username).exists():
        errors['username'] = 'This username is already taken by another user.'

    if not password:
        errors['password'] = 'Password is required.'
    else:
        pw_err = _validate_password(password)
        if pw_err:
            errors['password'] = pw_err
        elif password != confirm:
            errors['confirm_password'] = 'Passwords do not match.'

    email_err = _validate_email(email)
    if email_err:
        errors['email'] = email_err
    elif UserProfile.objects.filter(email=email).exists():
        errors['email'] = 'This email is already registered.'

    if not contact:
        errors['contact'] = 'Phone number is required.'
    elif UserProfile.objects.filter(contact=contact).exists():
        errors['contact'] = 'This phone number is already registered. Please choose another.'

    if errors:
        return render(request, 'register.html',
                      {'errors': errors, 'form_data': request.POST})

    UserProfile.objects.create(
        username=username,
        password=_hash_password(password),
        email=email,
        contact=contact,
        address=address,
    )
    return render(request, 'register.html',
                  {'success': True, 'registered_name': username})


# ── Dashboard ─────────────────────────────────────────────────────────────────

def dashboard(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')

    all_results = SegmentationResult.objects.filter(user=user).order_by('-uploaded_at')
    results = all_results[:20]

    total     = all_results.count()
    tumor_cnt = all_results.filter(has_tumor=True).count()
    avg_acc   = sum(r.accuracy for r in all_results) / total if total else 0

    otsu_results = all_results.filter(method='OTSU')
    dbim_results = all_results.filter(method='DBIM')
    avg_otsu = sum(r.accuracy for r in otsu_results) / otsu_results.count() if otsu_results.count() else 0
    avg_dbim = sum(r.accuracy for r in dbim_results) / dbim_results.count() if dbim_results.count() else 0

    context = {
        'user'       : user,
        'results'    : results,
        'total'      : total,
        'tumor_cnt'  : tumor_cnt,
        'avg_acc'    : round(avg_acc * 100, 1),
        'avg_otsu'   : round(avg_otsu * 100, 1),
        'avg_dbim'   : round(avg_dbim * 100, 1),
        'engine_ok'  : ENGINE_AVAILABLE,
    }
    return render(request, 'dashboard.html', context)


# ── OTSU Segmentation ─────────────────────────────────────────────────────────

def otsu_page(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')
    return render(request, 'otsu.html', {'user': user, 'engine_ok': ENGINE_AVAILABLE})


def otsu_action(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')
    if request.method != 'POST':
        return redirect('otsu')

    if 'image' not in request.FILES:
        return render(request, 'otsu.html',
                      {'user': user, 'error': 'Please upload an image.',
                       'engine_ok': ENGINE_AVAILABLE})

    img_file = request.FILES['image']
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
    fname = fs.save(img_file.name, img_file)
    fpath = os.path.join(settings.MEDIA_ROOT, 'uploads', fname)

    try:
        image_bgr = cv2.imread(fpath)
        if image_bgr is None:
            raise ValueError("Could not read image. Please upload a valid MRI/CT scan.")

        if not ENGINE_AVAILABLE:
            raise RuntimeError("Segmentation engine not available.")

        result = otsu_segment(image_bgr)

        orig_b64    = encode_image_b64(result['original'])
        seg_b64     = encode_image_b64(result['segmented'])
        overlay_b64 = encode_image_b64(result['overlay'])

        db_result = SegmentationResult.objects.create(
            user       = user,
            method     = 'OTSU',
            accuracy   = result['accuracy'],
            probability= result['probability'],
            has_tumor  = result['has_tumor'],
            otsu_thresh= result['otsu_val'],
            image_name = img_file.name,
        )

        features = result['features']
        params   = result['params']

        context = {
            'user'        : user,
            'orig_b64'    : orig_b64,
            'seg_b64'     : seg_b64,
            'overlay_b64' : overlay_b64,
            'accuracy'    : round(result['accuracy'] * 100, 2),
            'probability' : round(result['probability'] * 100, 2),
            'has_tumor'   : result['has_tumor'],
            'otsu_val'    : round(result['otsu_val'], 1),
            'img_mean'    : round(features['mean'], 1),
            'img_std'     : round(features['std'], 1),
            'img_entropy' : round(features['hist_entropy'], 3),
            'blur_ksize'  : params['blur_ksize'],
            'morph_w'     : params['morph_w'],
            'morph_h'     : params['morph_h'],
            'result_id'   : db_result.pk,
            'engine_ok'   : ENGINE_AVAILABLE,
        }
        return render(request, 'otsu.html', context)

    except Exception as e:
        return render(request, 'otsu.html',
                      {'user': user, 'error': str(e),
                       'engine_ok': ENGINE_AVAILABLE})
    finally:
        try:
            os.remove(fpath)
        except Exception:
            pass


# ── DBIM Segmentation ─────────────────────────────────────────────────────────

def dbim_page(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')
    return render(request, 'dbim.html', {'user': user, 'engine_ok': ENGINE_AVAILABLE})


def dbim_action(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')
    if request.method != 'POST':
        return redirect('dbim')

    if 'image' not in request.FILES:
        return render(request, 'dbim.html',
                      {'user': user, 'error': 'Please upload an image.',
                       'engine_ok': ENGINE_AVAILABLE})

    img_file = request.FILES['image']
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
    fname = fs.save(img_file.name, img_file)
    fpath = os.path.join(settings.MEDIA_ROOT, 'uploads', fname)

    try:
        image_bgr = cv2.imread(fpath)
        if image_bgr is None:
            raise ValueError("Could not read image. Please upload a valid MRI/CT scan.")

        if not ENGINE_AVAILABLE:
            raise RuntimeError("Segmentation engine not available.")

        result = dbim_segment(image_bgr)

        orig_b64    = encode_image_b64(result['original'])
        enh_b64     = encode_image_b64(result['enhanced'])
        tumor_b64   = encode_image_b64(result['tumor_image'])
        seg_b64     = encode_image_b64(result['segmented'])
        edges_b64   = encode_gray_b64(result['edges'])

        # Generate Otsu-thresholded image for side-by-side comparison
        gray_orig = cv2.cvtColor(
            cv2.resize(result['original'], (256, 256)),
            cv2.COLOR_BGR2GRAY
        )
        _, otsu_thresh_img = cv2.threshold(
            gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_thresh_bgr = cv2.cvtColor(otsu_thresh_img, cv2.COLOR_GRAY2BGR)
        otsu_thresh_b64 = encode_image_b64(otsu_thresh_bgr)

        db_result = SegmentationResult.objects.create(
            user        = user,
            method      = 'DBIM',
            accuracy    = result['accuracy'],
            probability = result['probability'],
            has_tumor   = result['has_tumor'],
            improvement = result['improvement'],
            image_name  = img_file.name,
        )

        features = result['features']
        params   = result['params']

        context = {
            'user'           : user,
            'orig_b64'       : orig_b64,
            'enh_b64'        : enh_b64,
            'tumor_b64'      : tumor_b64,
            'seg_b64'        : seg_b64,
            'edges_b64'      : edges_b64,
            'otsu_thresh_b64': otsu_thresh_b64,
            'accuracy'       : round(result['accuracy'] * 100, 2),
            'otsu_ref_acc'   : round(result['otsu_ref_acc'] * 100, 2),
            'improvement'    : round(result['improvement'] * 100, 2),
            'probability'    : round(result['probability'] * 100, 2),
            'has_tumor'      : result['has_tumor'],
            'canny_lo'       : params['canny_lo'],
            'canny_hi'       : params['canny_hi'],
            'dbim_thresh'    : params['dbim_thresh'],
            'erode_iters'    : params['erode_iters'],
            'dilate_iters'   : params['dilate_iters'],
            'img_mean'       : round(features['mean'], 1),
            'img_std'        : round(features['std'], 1),
            'gradient_mean'  : round(features['gradient_mean'], 1),
            'bbox_info'      : json.dumps(result['bbox_info']),
            'bbox_list'      : result['bbox_info'],
            'bbox_count'     : len(result['bbox_info']),
            'result_id'      : db_result.pk,
            'engine_ok'      : ENGINE_AVAILABLE,
        }
        return render(request, 'dbim.html', context)

    except Exception as e:
        return render(request, 'dbim.html',
                      {'user': user, 'error': str(e),
                       'engine_ok': ENGINE_AVAILABLE})
    finally:
        try:
            os.remove(fpath)
        except Exception:
            pass


# ── Accuracy Report ───────────────────────────────────────────────────────────

def accuracy_page(request):
    user = _get_current_user(request)
    if not user:
        return redirect('login')

    results = SegmentationResult.objects.filter(user=user)\
                                        .order_by('-uploaded_at')

    otsu_list = list(results.filter(method='OTSU').values(
        'accuracy', 'probability', 'has_tumor', 'uploaded_at', 'image_name'))
    dbim_list = list(results.filter(method='DBIM').values(
        'accuracy', 'probability', 'has_tumor', 'improvement',
        'uploaded_at', 'image_name'))

    def _avg(lst, key):
        vals = [d[key] for d in lst if d[key] is not None]
        return round(sum(vals) / len(vals) * 100, 2) if vals else 0

    context = {
        'user'           : user,
        'otsu_list'      : otsu_list,
        'dbim_list'      : dbim_list,
        'avg_otsu_acc'   : _avg(otsu_list, 'accuracy'),
        'avg_dbim_acc'   : _avg(dbim_list, 'accuracy'),
        'avg_improvement': _avg(dbim_list, 'improvement') if dbim_list else 0,
        'total_scans'    : len(otsu_list) + len(dbim_list),
        'otsu_json'      : json.dumps([
            {'x': i+1, 'y': round(d['accuracy']*100, 2)}
            for i, d in enumerate(otsu_list)]),
        'dbim_json'      : json.dumps([
            {'x': i+1, 'y': round(d['accuracy']*100, 2)}
            for i, d in enumerate(dbim_list)]),
    }
    return render(request, 'accuracy.html', context)
