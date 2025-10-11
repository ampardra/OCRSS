
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# اگر نیاز بود مسیر tesseract را برای ویندوز مشخص کنید:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_image_any(path: Path):
    ext = path.suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img
    # تلاش برای پردازش PDF در صورت وجود کتابخانه pdf2image
    if ext == '.pdf':
        try:
            from pdf2image import convert_from_path
        except Exception as e:
            raise RuntimeError("برای خواندن PDF نیاز به pdf2image داریم. نصب کنید: pip install pdf2image\n" + str(e))
        pages = convert_from_path(str(path))
        # تبدیل اولین صفحه به numpy array BGR
        pil = pages[0].convert('RGB')
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    raise ValueError(f"پسوند فایل پشتیبانی نمی‌شود: {ext}")

def resize_for_ocr(img, max_dim=2000):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def deskew_image(gray):
    """اصلاح کجی با استفاده از محاسبه زاویه با لحظه‌ها یا Hough"""
    coords = np.column_stack(np.where(gray < 255))  # متن تیره روی زمینه روشن
    if coords.size == 0:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess_for_ocr(bgr_img, enhance_text=True):
    """پیش‌پردازش پیشنهادی:
    - تبدیل به خاکستری
    - CLAHE (تقویت محلی کنتراست)
    - کاهش نویز با bilateral یا median
    - adaptive threshold یا Otsu
    - مورفولوژی برای پاکسازی نویزهای کوچک
    """
    img = bgr_img.copy()
    img, scale = resize_for_ocr(img, max_dim=2200)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE برای افزایش کنتراست محلی
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # کاهش نویز ملایم
    gray = cv2.medianBlur(gray, 3)

    # باینری‌سازی (سعی می‌کنیم بسته به تصویر یکی را انتخاب کنیم)
    try:
        # adaptive برای اسناد با روشنایی غیر یکنواخت
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # مورفولوژی پاکسازی نویز نقطه‌ای
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # معکوس برای deskew (بر پایه پیکسل‌های متن = سیاه)
    inv = 255 - opening

    # deskew روی تصویر باینری معکوس
    try:
        deskewed, angle = deskew_image(inv)
        deskewed = 255 - deskewed  # برگردانیم به فرمت متن تیره روی زمینه روشن
    except Exception:
        deskewed = opening
        angle = 0.0

    return deskewed, scale, angle

def ocr_with_data(img, lang='fas+eng', psm=3, oem=1, config_extra=''):
    """اجرای pytesseract و دریافت خروجی با جعبه‌ها و confidences"""
    pil = Image.fromarray(img)
    custom_config = f'--oem {oem} --psm {psm} {config_extra}'
    data = pytesseract.image_to_data(pil, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT)
    # خروجی dict شامل کلمات، جعبه‌ها و confidences است
    text = pytesseract.image_to_string(pil, lang=lang, config=custom_config)
    return text, data

def draw_boxes_and_save(orig_bgr, data, out_path:Path, scale=1.0):
    """رسم جعبه‌ها از data و ذخیره‌ی تصویر بررسی بصری"""
    img = orig_bgr.copy()
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i].lstrip('-').isdigit() else -1
        if txt == "" or conf <= 0:
            continue
        x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        draw.rectangle([x, y, x+w_box, y+h_box], outline="red", width=1)
        draw.text((x, max(0, y-16)), f"{txt} ({conf})", font=font)
    out_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_img)

def build_json_output(text, data, scale, angle):
    """ساختاردهی خروجی JSON"""
    items = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if txt == "" :
            continue
        conf = int(data['conf'][i]) if str(data['conf'][i]).lstrip('-').isdigit() else None
        items.append({
            'text': txt,
            'conf': conf,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'level': int(data['level'][i]) if 'level' in data else None
        })
    return {
        'full_text': text,
        'scale': scale,
        'deskew_angle': angle,
        'items': items
    }

def save_hocr(pil_img, lang, custom_config, out_path:Path):
    """ذخیره hOCR (برای حفظ layout)"""
    hocr = pytesseract.image_to_pdf_or_hocr(pil_img, lang=lang, config=custom_config, extension='hocr')
    with open(out_path, 'wb') as f:
        f.write(hocr)

def main():
    parser = argparse.ArgumentParser(description="OCR پیشرفته - پشتیبانی فارسی و خروجی JSON/hOCR/تصویر با جعبه‌ها")
    parser.add_argument('input', type=str, help='مسیر تصویر یا PDF')
    parser.add_argument('--lang', type=str, default='fas+eng', help="زبان برای tesseract (مثال: 'fas' یا 'eng' یا 'fas+eng')")
    parser.add_argument('--psm', type=int, default=3, help='Page segmentation mode')
    parser.add_argument('--oem', type=int, default=1, help='OCR Engine mode')
    parser.add_argument('--outdir', type=str, default=None, help='پوشه خروجی (پیش‌فرض همان مسیر تصویر)')
    parser.add_argument('--no-vis', action='store_true', help='عدم تولید تصویر با جعبه‌ها')
    parser.add_argument('--hocr', action='store_true', help='ذخیره خروجی hOCR')
    parser.add_argument('--config-extra', type=str, default='', help='پارامتر اضافی به tesseract (مثل -c tessedit_char_whitelist=...)')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print("فایل یافت نشد:", inp)
        sys.exit(1)

    try:
        orig_bgr = read_image_any(inp)
    except Exception as e:
        print("خطا در خواندن تصویر/PDF:", e)
        sys.exit(1)

    processed, scale, angle = preprocess_for_ocr(orig_bgr)
    text, data = ocr_with_data(processed, lang=args.lang, psm=args.psm, oem=args.oem, config_extra=args.config_extra)

    # تعیین مسیر خروجی
    out_base = Path(args.outdir) if args.outdir else inp.parent
    out_base.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    txt_path = out_base / f"{stem}.txt"
    json_path = out_base / f"{stem}.json"
    boxes_path = out_base / f"{stem}.boxes.jpg"
    hocr_path = out_base / f"{stem}.hocr"

    # ذخیره متن
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # ساخت و ذخیره JSON
    j = build_json_output(text, data, scale, angle)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(j, f, ensure_ascii=False, indent=2)

    # تصویر با جعبه‌ها
    if not args.no_vis:
        try:
            draw_boxes_and_save(orig_bgr, data, boxes_path, scale=scale)
        except Exception as e:
            print("خطا در تولید تصویر با جعبه‌ها:", e)

    # hOCR
    if args.hocr:
        try:
            pil = Image.fromarray(processed)
            custom_config = f'--oem {args.oem} --psm {args.psm} {args.config_extra}'
            save_hocr(pil, args.lang, custom_config, hocr_path)
        except Exception as e:
            print("خطا در تولید hOCR:", e)

    print("انجام شد.")
    print("خروجی‌ها:")
    print("  متن:", txt_path)
    print("  JSON:", json_path)
    if not args.no_vis:
        print("  تصویر جعبه‌ها:", boxes_path)
    if args.hocr:
        print("  hOCR:", hocr_path)

if __name__ == '__main__':
    main()
