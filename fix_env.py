#!/usr/bin/env python3
"""
สคริปต์แก้ไขไฟล์ .env ให้ comment ถูกต้อง
"""

import os
import shutil
from datetime import datetime

env_file = ".env"
backup_file = f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not os.path.exists(env_file):
    print(f"❌ ไม่พบไฟล์ {env_file}")
    exit(1)

# สำรองไฟล์เดิม
shutil.copy(env_file, backup_file)
print(f"✅ สำรองไฟล์เป็น: {backup_file}")

# อ่านไฟล์
with open(env_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# แก้ไขบรรทัดที่มี "Google Sheets Configuration" (อาจเป็นบรรทัด 8 หรือ 9)
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "Google Sheets Configuration" or (stripped.startswith("Google Sheets Configuration") and not stripped.startswith("#")):
        lines[i] = "# Google Sheets Configuration\n"
        print(f"✅ แก้ไขบรรทัดที่ {i+1}: {stripped} -> # Google Sheets Configuration")
        break

# เขียนไฟล์ใหม่
with open(env_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"✅ แก้ไขไฟล์ {env_file} เสร็จสิ้น")
print("\n📋 ตรวจสอบการแก้ไข (บรรทัด 7-9):")
for i in range(6, min(9, len(lines))):
    print(f"  {i+1}: {lines[i].rstrip()}")
