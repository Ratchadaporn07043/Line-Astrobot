# คู่มือการติดตั้ง Camelot

Camelot เป็น library สำหรับ extract ตารางจาก PDF ที่แม่นยำกว่า pdfplumber

## 📋 ความต้องการของระบบ

- Python 3.7+
- Ghostscript (สำหรับ PDF processing)
- OpenCV (ถ้าต้องการใช้ `camelot-py[cv]`)

---

## 🍎 สำหรับ macOS

### 1. ติดตั้ง Homebrew (ถ้ายังไม่มี)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. ติดตั้ง Ghostscript

```bash
brew install ghostscript
```

### 3. (ถ้าจำเป็น) สร้าง symbolic link สำหรับ Ghostscript

ถ้าเจอปัญหา "Ghostscript module cannot be found":

```bash
mkdir -p ~/lib
ln -s "$(brew --prefix gs)/lib/libgs.dylib" ~/lib
```

### 4. ติดตั้ง Camelot

```bash
# ติดตั้งพร้อม OpenCV support (แนะนำ)
pip install "camelot-py[cv]"

# หรือติดตั้งแบบพื้นฐาน
pip install "camelot-py[base]"
```

### 5. ตรวจสอบการติดตั้ง

```python
python3 -c "from ctypes.util import find_library; print(find_library('gs'))"
```

ถ้าได้ path กลับมา แสดงว่าติดตั้งสำเร็จ ✅

---

## 🐧 สำหรับ Linux (Ubuntu/Debian)

### 1. ติดตั้ง dependencies

```bash
sudo apt-get update
sudo apt-get install ghostscript python3-tk
```

### 2. ติดตั้ง Camelot

```bash
# ติดตั้งพร้อม OpenCV support (แนะนำ)
pip install "camelot-py[cv]"

# หรือติดตั้งแบบพื้นฐาน
pip install "camelot-py[base]"
```

### 3. ตรวจสอบการติดตั้ง

```python
python3 -c "from ctypes.util import find_library; print(find_library('gs'))"
```

---

## 🪟 สำหรับ Windows

### 1. ดาวน์โหลดและติดตั้ง Ghostscript

1. ดาวน์โหลดจาก: https://www.ghostscript.com/download/gsdnld.html
2. ติดตั้งตามปกติ
3. เพิ่ม Ghostscript ไปยัง PATH environment variable

### 2. ติดตั้ง Camelot

```bash
# ติดตั้งพร้อม OpenCV support (แนะนำ)
pip install "camelot-py[cv]"

# หรือติดตั้งแบบพื้นฐาน
pip install "camelot-py[base]"
```

---

## ✅ ทดสอบการติดตั้ง

สร้างไฟล์ทดสอบ `test_camelot.py`:

```python
#!/usr/bin/env python3
"""ทดสอบการติดตั้ง Camelot"""

try:
    import camelot
    print("✅ Camelot ติดตั้งสำเร็จ!")
    print(f"   Version: {camelot.__version__}")
    
    # ทดสอบ import dependencies
    from ctypes.util import find_library
    gs_path = find_library('gs')
    if gs_path:
        print(f"✅ Ghostscript พบที่: {gs_path}")
    else:
        print("⚠️ Ghostscript ไม่พบ - อาจมีปัญหาในการใช้งาน")
        
except ImportError as e:
    print(f"❌ Camelot ยังไม่ได้ติดตั้ง: {e}")
    print("   กรุณาติดตั้งด้วย: pip install 'camelot-py[cv]'")
```

รันทดสอบ:

```bash
python3 test_camelot.py
```

---

## 🔧 แก้ไขปัญหา

### ปัญหา: "Ghostscript not found"

**macOS:**
```bash
brew install ghostscript
mkdir -p ~/lib
ln -s "$(brew --prefix gs)/lib/libgs.dylib" ~/lib
```

**Linux:**
```bash
sudo apt-get install ghostscript
```

**Windows:**
- ตรวจสอบว่า Ghostscript ถูกติดตั้งแล้ว
- ตรวจสอบว่า PATH มี Ghostscript bin directory

### ปัญหา: "No module named 'camelot'"

```bash
pip install "camelot-py[cv]"
```

### ปัญหา: ImportError สำหรับ OpenCV

```bash
pip install opencv-python
```

---

## 📚 เอกสารเพิ่มเติม

- [Camelot Documentation](https://camelot-py.readthedocs.io/)
- [Installation Guide](https://camelot-py.readthedocs.io/en/master/user/install.html)
- [Dependencies](https://camelot-py.readthedocs.io/en/master/user/install-deps.html)

---

## 💡 หมายเหตุ

- Camelot ใช้ pandas DataFrame อยู่แล้ว แต่เราไม่ต้อง import pandas ในโค้ดของเรา
- ระบบจะ fallback เป็น pdfplumber ถ้า Camelot ไม่ได้ติดตั้ง
- สำหรับตารางที่มีเส้นขอบ ใช้ `flavor='lattice'`
- สำหรับตารางที่ไม่มีเส้นขอบ ใช้ `flavor='stream'`
