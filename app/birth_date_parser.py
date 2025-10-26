import re
from datetime import datetime
import logging
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from .astronomical_calculator import AstronomicalCalculator


# ตั้งค่า logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# โหลด environment variables
load_dotenv()

class BirthDateParser:
    """Class สำหรับแปลงวันเกิดจากข้อความในรูปแบบต่างๆ"""
    
    def __init__(self):
        # สร้างเครื่องคำนวณดาราศาสตร์
        self.astronomical_calculator = AstronomicalCalculator()
        
        # Dictionary สำหรับแปลงชื่อเดือนไทยเป็นตัวเลข
        self.thai_months = {
            'มกราคม': 1, 'ม.ค.': 1, 'มค': 1,
            'กุมภาพันธ์': 2, 'ก.พ.': 2, 'กพ': 2,
            'มีนาคม': 3, 'มี.ค.': 3, 'มีค': 3,
            'เมษายน': 4, 'เม.ย.': 4, 'เมย': 4,
            'พฤษภาคม': 5, 'พ.ค.': 5, 'พค': 5,
            'มิถุนายน': 6, 'มิ.ย.': 6, 'มิย': 6,
            'กรกฎาคม': 7, 'ก.ค.': 7, 'กค': 7,
            'สิงหาคม': 8, 'ส.ค.': 8, 'สค': 8,
            'กันยายน': 9, 'ก.ย.': 9, 'กย': 9,
            'ตุลาคม': 10, 'ต.ค.': 10, 'ตค': 10,
            'พฤศจิกายน': 11, 'พ.ย.': 11, 'พย': 11,
            'ธันวาคม': 12, 'ธ.ค.': 12, 'ธค': 12
        }
        
        # Dictionary สำหรับแปลงชื่อเดือนอังกฤษเป็นตัวเลข
        self.english_months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # Pattern ต่างๆ สำหรับ regex
        self.patterns = [
            # รูปแบบ dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy (รองรับข้อความติดกัน)
            (r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})', 'dmy'),
            
            # รูปแบบ yyyy/mm/dd, yyyy-mm-dd (รองรับข้อความติดกัน)
            (r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})', 'ymd'),
            
            # รูปแบบ dd mm yyyy (เว้นวรรค)
            (r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})\b', 'dmy'),
            
            # รูปแบบ วันที่ X เดือน Y ปี Z
            (r'วันที่\s*(\d{1,2})\s*เดือน\s*(\d{1,2})\s*ปี\s*(\d{2,4})', 'dmy'),
            
            # รูปแบบ เกิดวันที่ X/Y/Z
            (r'เกิด.*?(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', 'dmy'),
            
            # รูปแบบ วันเกิด X/Y/Z
            (r'วันเกิด.*?(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})', 'dmy'),
            
            # รูปแบบ เดือนชื่อไทย เช่น 7 มกราคม 2003
            (r'(\d{1,2})\s*(' + '|'.join(self.thai_months.keys()) + r')\s*(\d{2,4})', 'thai_month'),
            
            # รูปแบบ เดือนชื่ออังกฤษ เช่น 7 January 2003
            (r'(\d{1,2})\s*(' + '|'.join(self.english_months.keys()) + r')\s*(\d{2,4})', 'english_month'),
            
            # รูปแบบ แค่ตัวเลข 8 หลัก เช่น 07092003 (รองรับข้อความติดกัน)
            (r'(\d{2})(\d{2})(\d{4})', 'ddmmyyyy'),
            
            # รูปแบบ ปีไทย (พ.ศ.) เช่น 7/9/2546
            (r'(\d{1,2})[\/\-](\d{1,2})[\/\-](25\d{2})', 'thai_year'),
        ]
        
        # Pattern สำหรับเวลาเกิด
        self.time_patterns = [
            # รูปแบบ HH:MM เช่น 14:30, 2:30
            (r'(\d{1,2}):(\d{2})', 'time'),
            # รูปแบบ HH.MM เช่น 14.30, 2.30
            (r'(\d{1,2})\.(\d{2})', 'time'),
            # รูปแบบ HH MM เช่น 14 30, 2 30
            (r'(\d{1,2})\s+(\d{2})', 'time'),
            # รูปแบบ เวลา X นาฬิกา Y นาที
            (r'เวลา\s*(\d{1,2})\s*นาฬิกา\s*(\d{1,2})\s*นาที', 'time'),
            # รูปแบบ X นาฬิกา Y นาที
            (r'(\d{1,2})\s*นาฬิกา\s*(\d{1,2})\s*นาที', 'time'),
            # รูปแบบ X น. Y นาที
            (r'(\d{1,2})\s*น\.\s*(\d{1,2})\s*นาที', 'time'),
        ]
        
        # Dictionary สำหรับสถานที่เกิดและพิกัด
        self.location_coordinates = {
            # เมืองใหญ่ในประเทศไทย
            'กรุงเทพ': {'lat': 13.7563, 'lon': 100.5018},
            'กรุงเทพฯ': {'lat': 13.7563, 'lon': 100.5018},
            'กรุงเทพมหานคร': {'lat': 13.7563, 'lon': 100.5018},
            'เชียงใหม่': {'lat': 18.7883, 'lon': 98.9853},
            'เชียงราย': {'lat': 19.9105, 'lon': 99.8405},
            'นครราชสีมา': {'lat': 14.9799, 'lon': 102.0978},
            'ขอนแก่น': {'lat': 16.4419, 'lon': 102.8359},
            'อุดรธานี': {'lat': 17.4138, 'lon': 102.7873},
            'อุบลราชธานี': {'lat': 15.2287, 'lon': 104.8563},
            'สงขลา': {'lat': 7.0061, 'lon': 100.5008},
            'ภูเก็ต': {'lat': 7.8804, 'lon': 98.3923},
            'พัทยา': {'lat': 12.9236, 'lon': 100.8825},
            'หัวหิน': {'lat': 12.5684, 'lon': 99.9576},
            'สุราษฎร์ธานี': {'lat': 9.1382, 'lon': 99.3215},
            'นครศรีธรรมราช': {'lat': 8.4304, 'lon': 99.9631},
            'ยะลา': {'lat': 6.5414, 'lon': 101.2804},
            'ปัตตานี': {'lat': 6.8694, 'lon': 101.2503},
            'นราธิวาส': {'lat': 6.4255, 'lon': 101.8253},
            'ระยอง': {'lat': 12.6819, 'lon': 101.2819},
            'ชลบุรี': {'lat': 13.3611, 'lon': 100.9847},
            'สมุทรปราการ': {'lat': 13.5991, 'lon': 100.5998},
            'นนทบุรี': {'lat': 13.8668, 'lon': 100.5168},
            'ปทุมธานี': {'lat': 14.0208, 'lon': 100.5250},
            'นครปฐม': {'lat': 13.8199, 'lon': 100.0623},
            'ราชบุรี': {'lat': 13.5360, 'lon': 99.8134},
            'กาญจนบุรี': {'lat': 14.0228, 'lon': 99.5328},
            'สุพรรณบุรี': {'lat': 14.4745, 'lon': 100.1226},
            'อ่างทอง': {'lat': 14.5896, 'lon': 100.4550},
            'ลพบุรี': {'lat': 14.7995, 'lon': 100.6534},
            'สิงห์บุรี': {'lat': 14.8936, 'lon': 100.3969},
            'ชัยนาท': {'lat': 15.1855, 'lon': 100.1251},
            'อุทัยธานี': {'lat': 15.3795, 'lon': 99.5089},
            'กำแพงเพชร': {'lat': 16.4828, 'lon': 99.5227},
            'ตาก': {'lat': 16.8845, 'lon': 98.8565},
            'สุโขทัย': {'lat': 17.0056, 'lon': 99.8262},
            'พิษณุโลก': {'lat': 16.8211, 'lon': 100.2659},
            'พิจิตร': {'lat': 16.4388, 'lon': 100.3488},
            'เพชรบูรณ์': {'lat': 16.4190, 'lon': 101.1606},
            'ลำปาง': {'lat': 18.2980, 'lon': 99.4909},
            'ลำพูน': {'lat': 18.5801, 'lon': 99.0078},
            'แม่ฮ่องสอน': {'lat': 19.3019, 'lon': 97.9651},
            'น่าน': {'lat': 18.7756, 'lon': 100.7730},
            'พะเยา': {'lat': 19.1920, 'lon': 99.9016},
            'แพร่': {'lat': 18.1449, 'lon': 100.1406},
            'นครสวรรค์': {'lat': 15.7047, 'lon': 100.1371},
            'อุตรดิตถ์': {'lat': 17.6201, 'lon': 100.0993},
            'กาฬสินธุ์': {'lat': 16.4419, 'lon': 103.5060},
            'สกลนคร': {'lat': 17.1536, 'lon': 104.1409},
            'นครพนม': {'lat': 17.4074, 'lon': 104.7789},
            'มุกดาหาร': {'lat': 16.5453, 'lon': 104.7235},
            'ร้อยเอ็ด': {'lat': 16.0538, 'lon': 103.6530},
            'ยโสธร': {'lat': 15.7924, 'lon': 104.1453},
            'อำนาจเจริญ': {'lat': 15.8650, 'lon': 104.6258},
            'หนองบัวลำภู': {'lat': 17.2218, 'lon': 102.4447},
            'เลย': {'lat': 17.4860, 'lon': 101.7223},
            'หนองคาย': {'lat': 17.8783, 'lon': 102.7413},
            'มหาสารคาม': {'lat': 16.1844, 'lon': 103.3020},
            'สุรินทร์': {'lat': 14.8826, 'lon': 103.4938},
            'ศรีสะเกษ': {'lat': 15.1186, 'lon': 104.3220},
            'บุรีรัมย์': {'lat': 14.9932, 'lon': 103.1029},
            'ชัยภูมิ': {'lat': 15.8067, 'lon': 102.0313},
            'เพชรบุรี': {'lat': 13.1119, 'lon': 99.9447},
            'ประจวบคีรีขันธ์': {'lat': 11.8124, 'lon': 99.7979},
            'ชุมพร': {'lat': 10.4930, 'lon': 99.1800},
            'ระนอง': {'lat': 9.9658, 'lon': 98.6347},
            'กระบี่': {'lat': 8.0863, 'lon': 98.9063},
            'ตรัง': {'lat': 7.5567, 'lon': 99.6114},
            'พังงา': {'lat': 8.4505, 'lon': 98.5319},
            'สตูล': {'lat': 6.6238, 'lon': 100.0674},
            'นครนายก': {'lat': 14.2069, 'lon': 101.2131},
            'สระแก้ว': {'lat': 13.8240, 'lon': 102.0644},
            'สระบุรี': {'lat': 14.5289, 'lon': 100.9101},
            'ตราด': {'lat': 12.2436, 'lon': 102.5150},
            'จันทบุรี': {'lat': 12.6117, 'lon': 102.1038},
            'ฉะเชิงเทรา': {'lat': 13.6904, 'lon': 101.0779},
            'ปราจีนบุรี': {'lat': 14.0507, 'lon': 101.3703},
            'สมุทรสาคร': {'lat': 13.5991, 'lon': 100.2744},
            'สมุทรสงคราม': {'lat': 13.4149, 'lon': 100.0026},
            'นครนายก': {'lat': 14.2069, 'lon': 101.2131},
            'พระนครศรีอยุธยา': {'lat': 14.3692, 'lon': 100.5877},
            'อยุธยา': {'lat': 14.3692, 'lon': 100.5877},
        }

    def extract_birth_date(self, text: str) -> str:
        """
        แยกวันเกิดจากข้อความ
        
        Args:
            text (str): ข้อความที่ต้องการแยกวันเกิด
            
        Returns:
            str: วันเกิดในรูปแบบ dd/mm/yyyy หรือ None ถ้าไม่พบ
        """
        text = text.lower().strip()
        logger.info(f"กำลังแยกวันเกิดจาก: {text}")
        
        for pattern, format_type in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if matches:
                match = matches[0]
                logger.info(f"พบ pattern {format_type}: {match}")
                
                try:
                    birth_date = self._parse_match(match, format_type)
                    if birth_date:
                        logger.info(f"แปลงวันเกิดสำเร็จ: {birth_date}")
                        return birth_date
                except Exception as e:
                    logger.warning(f"แปลงวันเกิดไม่สำเร็จ: {e}")
                    continue
        
        logger.warning("ไม่พบวันเกิดในข้อความ")
        return None

    def extract_birth_time(self, text: str) -> str:
        """
        แยกเวลาเกิดจากข้อความ
        
        Args:
            text (str): ข้อความที่ต้องการแยกเวลาเกิด
            
        Returns:
            str: เวลาเกิดในรูปแบบ HH:MM หรือ None ถ้าไม่พบ
        """
        text = text.lower().strip()
        logger.info(f"กำลังแยกเวลาเกิดจาก: {text}")
        
        for pattern, format_type in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if matches:
                match = matches[0]
                logger.info(f"พบ time pattern {format_type}: {match}")
                
                try:
                    birth_time = self._parse_time_match(match, format_type)
                    if birth_time:
                        logger.info(f"แปลงเวลาเกิดสำเร็จ: {birth_time}")
                        return birth_time
                except Exception as e:
                    logger.warning(f"แปลงเวลาเกิดไม่สำเร็จ: {e}")
                    continue
        
        logger.warning("ไม่พบเวลาเกิดในข้อความ")
        return None

    def extract_birth_location(self, text: str) -> dict:
        """
        แยกสถานที่เกิดจากข้อความ
        
        Args:
            text (str): ข้อความที่ต้องการแยกสถานที่เกิด
            
        Returns:
            dict: ข้อมูลสถานที่เกิด {'location': str, 'latitude': float, 'longitude': float}
        """
        text_lower = text.lower().strip()
        logger.info(f"กำลังแยกสถานที่เกิดจาก: {text}")
        
        # ค้นหาสถานที่ในข้อความ
        for location_name, coordinates in self.location_coordinates.items():
            if location_name.lower() in text_lower:
                logger.info(f"พบสถานที่เกิด: {location_name}")
                return {
                    'location': location_name,
                    'latitude': coordinates['lat'],
                    'longitude': coordinates['lon']
                }
        
        # ไม่พบสถานที่เกิดในข้อความ ใช้กรุงเทพฯ เป็นค่าเริ่มต้น
        logger.info("ไม่พบสถานที่เกิดในข้อความ ใช้กรุงเทพฯ เป็นค่าเริ่มต้นสำหรับการคำนวณ Ascendant")
        return {
            'location': 'กรุงเทพฯ',
            'latitude': 13.7563,
            'longitude': 100.5018
        }

    def extract_birth_info(self, text: str) -> dict:
        """
        แยกข้อมูลวันเกิด เวลาเกิด และสถานที่เกิดจากข้อความ
        
        Args:
            text (str): ข้อความที่ต้องการแยกข้อมูล
            
        Returns:
            dict: ข้อมูลวันเกิด เวลาเกิด และสถานที่เกิด
        """
        birth_date = self.extract_birth_date(text)
        birth_time = self.extract_birth_time(text)
        birth_location = self.extract_birth_location(text)
        
        return {
            'date': birth_date,
            'time': birth_time,
            'location': birth_location['location'],
            'latitude': birth_location['latitude'],
            'longitude': birth_location['longitude']
        }

    def _parse_match(self, match, format_type):
        """แปลง match ให้เป็นวันเกิดในรูปแบบ dd/mm/yyyy"""
        
        if format_type == 'dmy':
            day, month, year = match
            return self._format_date(int(day), int(month), int(year))
            
        elif format_type == 'ymd':
            year, month, day = match
            return self._format_date(int(day), int(month), int(year))
            
        elif format_type == 'thai_month':
            day, month_name, year = match
            month = self.thai_months.get(month_name.lower())
            if month:
                return self._format_date(int(day), month, int(year))
                
        elif format_type == 'english_month':
            day, month_name, year = match
            month = self.english_months.get(month_name.lower())
            if month:
                return self._format_date(int(day), month, int(year))
                
        elif format_type == 'ddmmyyyy':
            day, month, year = match
            return self._format_date(int(day), int(month), int(year))
            
        elif format_type == 'thai_year':
            day, month, thai_year = match
            # แปลง พ.ศ. เป็น ค.ศ.
            year = int(thai_year) - 543
            return self._format_date(int(day), int(month), year)
        
        return None

    def _parse_time_match(self, match, format_type):
        """แปลง match ให้เป็นเวลาในรูปแบบ HH:MM"""
        if format_type == 'time':
            hour, minute = match
            hour = int(hour)
            minute = int(minute)
            
            # ตรวจสอบความถูกต้องของเวลา
            if not (0 <= hour <= 23):
                return None
            if not (0 <= minute <= 59):
                return None
            
            return f"{hour:02d}:{minute:02d}"
        
        return None

    def _format_date(self, day, month, year):
        """ตรวจสอบและจัดรูปแบบวันที่"""
        
        # ปรับปีให้เป็น 4 หลัก
        if year < 100:
            if year <= 30:  # 00-30 = 2000-2030
                year += 2000
            else:  # 31-99 = 1931-1999
                year += 1900
        
        # ตรวจสอบความถูกต้องของวันที่
        if not (1 <= month <= 12):
            return None
            
        if not (1 <= day <= 31):
            return None
            
        if not (1900 <= year <= datetime.now().year + 10):
            return None
        
        # ตรวจสอบวันที่ในเดือน
        try:
            datetime(year, month, day)
        except ValueError:
            return None
        
        # ส่งกลับในรูปแบบ dd/mm/yyyy
        return f"{day:02d}/{month:02d}/{year}"

    def calculate_zodiac_sign(self, day: int, month: int) -> dict:
        """
        คำนวณราศีจากวันและเดือน (Western Astrology)
        
        Args:
            day (int): วัน
            month (int): เดือน
            
        Returns:
            dict: ข้อมูลราศี {'sign': 'ชื่อราศี', 'element': 'ธาตุ', 'quality': 'คุณภาพ'}
        """
        # ราศีและข้อมูล
        zodiac_data = {
            'aries': {'name': 'เมษ', 'element': 'ไฟ', 'quality': 'Cardinal', 'dates': [(3, 21), (4, 19)]},
            'taurus': {'name': 'พฤษภ', 'element': 'ดิน', 'quality': 'Fixed', 'dates': [(4, 20), (5, 20)]},
            'gemini': {'name': 'เมถุน', 'element': 'ลม', 'quality': 'Mutable', 'dates': [(5, 21), (6, 20)]},
            'cancer': {'name': 'กรกฎ', 'element': 'น้ำ', 'quality': 'Cardinal', 'dates': [(6, 21), (7, 22)]},
            'leo': {'name': 'สิงห์', 'element': 'ไฟ', 'quality': 'Fixed', 'dates': [(7, 23), (8, 22)]},
            'virgo': {'name': 'กันย์', 'element': 'ดิน', 'quality': 'Mutable', 'dates': [(8, 23), (9, 22)]},
            'libra': {'name': 'ตุล', 'element': 'ลม', 'quality': 'Cardinal', 'dates': [(9, 23), (10, 22)]},
            'scorpio': {'name': 'พิจิก', 'element': 'น้ำ', 'quality': 'Fixed', 'dates': [(10, 23), (11, 21)]},
            'sagittarius': {'name': 'ธนู', 'element': 'ไฟ', 'quality': 'Mutable', 'dates': [(11, 22), (12, 21)]},
            'capricorn': {'name': 'มังกร', 'element': 'ดิน', 'quality': 'Cardinal', 'dates': [(12, 22), (1, 19)]},
            'aquarius': {'name': 'กุมภ์', 'element': 'ลม', 'quality': 'Fixed', 'dates': [(1, 20), (2, 18)]},
            'pisces': {'name': 'มีน', 'element': 'น้ำ', 'quality': 'Mutable', 'dates': [(2, 19), (3, 20)]}
        }
        
        # ค้นหาราศี
        for sign_key, sign_info in zodiac_data.items():
            start_month, start_day = sign_info['dates'][0]
            end_month, end_day = sign_info['dates'][1]
            
            # ตรวจสอบราศีมังกร (ข้ามปี)
            if sign_key == 'capricorn':
                if (month == 12 and day >= start_day) or (month == 1 and day <= end_day):
                    return {
                        'sign': sign_info['name'],
                        'element': sign_info['element'],
                        'quality': sign_info['quality'],
                        'english_name': sign_key.title()
                    }
            else:
                if (month == start_month and day >= start_day) or (month == end_month and day <= end_day):
                    return {
                        'sign': sign_info['name'],
                        'element': sign_info['element'],
                        'quality': sign_info['quality'],
                        'english_name': sign_key.title()
                    }
        
        return None

    def generate_birth_chart_info(self, birth_date: str, birth_time: str = None, latitude: float = 13.7563, longitude: float = 100.5018) -> dict:
        """
        สร้างข้อมูลดวงชะตาพื้นฐาน รวมถึงการคำนวณ Ascendant
        
        Args:
            birth_date (str): วันเกิดในรูปแบบ dd/mm/yyyy
            birth_time (str): เวลาเกิดในรูปแบบ HH:MM (ไม่บังคับ)
            latitude (float): ละติจูดของสถานที่เกิด (default: กรุงเทพฯ)
            longitude (float): ลองจิจูดของสถานที่เกิด (default: กรุงเทพฯ)
            
        Returns:
            dict: ข้อมูลดวงชะตาพื้นฐาน
        """
        if not birth_date:
            return None
        
        try:
            # แปลงวันเกิด
            day, month, year = map(int, birth_date.split('/'))
            
            # คำนวณราศี
            zodiac_info = self.calculate_zodiac_sign(day, month)
            
            if not zodiac_info:
                return None
            
            # สร้าง birth_datetime
            birth_datetime = datetime(year, month, day)
            
            # ถ้ามีเวลาเกิด ให้เพิ่มเข้าไปใน birth_datetime
            if birth_time:
                try:
                    hour, minute = map(int, birth_time.split(':'))
                    birth_datetime = birth_datetime.replace(hour=hour, minute=minute)
                except:
                    logger.warning(f"Invalid birth time format: {birth_time}")
            
            # คำนวณอายุ
            age = datetime.now().year - year
            
            # สร้างข้อมูลดวงชะตา
            chart_info = {
                'birth_date': birth_date,
                'birth_time': birth_time,
                'age': age,
                'zodiac_sign': zodiac_info['sign'],
                'zodiac_element': zodiac_info['element'],
                'zodiac_quality': zodiac_info['quality'],
                'zodiac_english': zodiac_info['english_name'],
                'birth_datetime': birth_datetime,
                'birth_location': {
                    'latitude': latitude,
                    'longitude': longitude
                }
            }
            
            # คำนวณ Ascendant ถ้ามีเวลาเกิด
            if birth_time:
                try:
                    ascendant_data = self.astronomical_calculator.calculate_ascendant(
                        birth_datetime, latitude, longitude
                    )
                    if ascendant_data:
                        chart_info['ascendant'] = ascendant_data
                        chart_info['ascendant_interpretation'] = self.astronomical_calculator.get_ascendant_interpretation(ascendant_data)
                        logger.info(f"✅ Calculated Ascendant: {ascendant_data['sign']} {ascendant_data['degree']:.1f}°")
                    else:
                        logger.warning("Failed to calculate Ascendant")
                except Exception as e:
                    logger.error(f"Error calculating Ascendant: {e}")
            
            # คำนวณบ้านทั้ง 12 บ้าน ถ้ามีเวลาเกิด
            if birth_time:
                try:
                    houses_data = self.astronomical_calculator.calculate_house_cusps(
                        birth_datetime, latitude, longitude
                    )
                    if houses_data:
                        chart_info['houses'] = houses_data
                        logger.info(f"✅ Calculated 12 houses")
                    else:
                        logger.warning("Failed to calculate houses")
                except Exception as e:
                    logger.error(f"Error calculating houses: {e}")
            
            return chart_info
            
        except Exception as e:
            logger.error(f"Error generating birth chart: {e}")
            return None

    def test_parser(self):
        """ทดสอบ parser ด้วยตัวอย่างต่างๆ"""
        test_cases = [
            "07/09/2003",
            "7-9-2003",
            "7.9.2003",
            "2003/09/07",
            "7 9 2003",
            "วันที่ 7 เดือน 9 ปี 2003",
            "วันที่7เดือน9ปี2003",
            "เกิดวันที่ 7/9/2003",
            "วันเกิดของฉันคือ 07/09/2003",
            "7 มกราคม 2003",
            "7 ม.ค. 2003",
            "7 January 2003",
            "7 Jan 2003",
            "07092003",
            "7/9/2546",  # ปี พ.ศ.
            "เกิด 15 พ.ค. 90",
            "15/05/90",
            "ฉันเกิดวันที่ 25 ธันวาคม 1985",
            "สวัสดีครับ วันเกิดผมคือ 15/03/1990 ครับ",
            "Hello my birthday is 15/03/1990",
            "07/09/2003ราศีอะไร",  # 🆕 ทดสอบข้อความติดกัน
            "15/03/1990ราศีอะไร",
            "ไม่มีวันเกิดในข้อความนี้",
            # 🆕 ทดสอบเวลาเกิด
            "เกิดวันที่ 7/9/2003 เวลา 14:30",
            "วันเกิด 15/03/1990 เวลา 2 นาฬิกา 30 นาที",
            "เกิด 25/12/1985 เวลา 8.30",
            "7/9/2003 14:30 ทำนายดวงชะตา",
            # 🆕 ทดสอบสถานที่เกิด
            "เกิดวันที่ 7/9/2003 เวลา 14:30 ที่เชียงใหม่",
            "วันเกิด 15/03/1990 เวลา 2 นาฬิกา 30 นาที ภูเก็ต",
            "เกิด 25/12/1985 เวลา 8.30 กรุงเทพฯ",
            "7/9/2003 14:30 เชียงใหม่ ทำนายดวงชะตา"
        ]
        
        print("🧪 ทดสอบ Birth Date Parser")
        print("=" * 50)
        
        for i, test in enumerate(test_cases, 1):
            result = self.extract_birth_date(test)
            time_result = self.extract_birth_time(test)
            location_result = self.extract_birth_location(test)
            status = "✅" if result else "❌"
            time_status = "⏰" if time_result else "⏸️"
            location_status = "📍" if location_result['location'] != 'กรุงเทพฯ' else "🏠"
            print(f"{i:2d}. {status} {time_status} {location_status} '{test}' → Date: {result}, Time: {time_result}, Location: {location_result['location']}")
            
            # ทดสอบการสร้างดวงชะตา
            if result:
                birth_info = self.generate_birth_chart_info(result, time_result, location_result['latitude'], location_result['longitude'])
                if birth_info:
                    print(f"    🌟 ราศี: {birth_info['zodiac_sign']} ({birth_info['zodiac_element']})")
                    if 'birth_location_name' in birth_info:
                        print(f"    📍 สถานที่เกิด: {birth_info['birth_location_name']}")
                    
                    # แสดงข้อมูล Ascendant ถ้ามี
                    if 'ascendant' in birth_info:
                        ascendant = birth_info['ascendant']
                        print(f"    🌅 Ascendant: ราศี{ascendant['sign']} {ascendant['degree']:.1f}° ({ascendant['element']})")
                        print(f"    📝 การตีความ: {birth_info.get('ascendant_interpretation', 'ไม่มีข้อมูล')}")
                    
                    # แสดงข้อมูลบ้านถ้ามี
                    if 'houses' in birth_info:
                        print(f"    🏠 บ้านทั้ง 12 บ้าน: คำนวณแล้ว")
                        # แสดงบ้านสำคัญ
                        important_houses = [1, 4, 7, 10]  # Ascendant, IC, Descendant, MC
                        for house_num in important_houses:
                            house_data = birth_info['houses'].get(f'house_{house_num}')
                            if house_data:
                                print(f"       บ้านที่ {house_num}: ราศี{house_data['sign']} {house_data['degree']:.1f}°")
                print()

# ฟังก์ชันหลักสำหรับใช้ใน response_message.py
def extract_birth_date_from_message(message: str) -> str:
    """
    ฟังก์ชันง่ายๆ สำหรับแยกวันเกิดจากข้อความ
    
    Args:
        message (str): ข้อความจากผู้ใช้
        
    Returns:
        str: วันเกิดในรูปแบบ dd/mm/yyyy หรือ None
    """
    parser = BirthDateParser()
    return parser.extract_birth_date(message)

def extract_birth_info_from_message(message: str) -> dict:
    """
    ฟังก์ชันสำหรับแยกข้อมูลวันเกิด เวลาเกิด และสถานที่เกิดจากข้อความ
    
    Args:
        message (str): ข้อความจากผู้ใช้
        
    Returns:
        dict: ข้อมูลวันเกิด เวลาเกิด และสถานที่เกิด
    """
    parser = BirthDateParser()
    return parser.extract_birth_info(message)

def get_zodiac_data_from_mongodb(zodiac_sign: str) -> dict:
    """
    ดึงข้อมูลราศีจาก MongoDB
    
    Args:
        zodiac_sign (str): ชื่อราศี
        
    Returns:
        dict: ข้อมูลราศี
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            logger.warning("MONGO_URL not configured properly")
            return None
            
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        db = client["astrobot"]
        collection = db["zodiac_personality"]
        
        # ค้นหาข้อมูลราศี
        zodiac_data = collection.find_one({"zodiac_sign": zodiac_sign})
        client.close()
        
        if zodiac_data:
            # แปลงข้อมูลให้ตรงกับรูปแบบเดิม
            return {
                "ลักษณะนิสัย": zodiac_data.get("personality_traits", ""),
                "การงาน": zodiac_data.get("career", ""),
                "การเงิน": zodiac_data.get("finance", ""),
                "สุขภาพ": zodiac_data.get("health", ""),
                "ความรัก": zodiac_data.get("love", {})
            }
        else:
            logger.warning(f"No data found for zodiac sign: {zodiac_sign}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching zodiac data from MongoDB: {e}")
        return None

def generate_astrology_reading(message: str) -> dict:
    """
    ฟังก์ชันสำหรับสร้างการทำนายดวงชะตาจากข้อความ
    
    Args:
        message (str): ข้อความจากผู้ใช้
        
    Returns:
        dict: ข้อมูลดวงชะตาพร้อมการทำนาย
    """
    parser = BirthDateParser()
    birth_info = parser.extract_birth_info(message)
    
    if not birth_info or not birth_info['date']:
        return None
    
    # สร้างข้อมูลดวงชะตา
    chart_info = parser.generate_birth_chart_info(birth_info['date'], birth_info['time'])
    
    if not chart_info:
        return None
    
    return chart_info


def generate_detailed_astrology_reading(message: str, latitude: float = None, longitude: float = None) -> dict:
    """
    ฟังก์ชันสำหรับสร้างการทำนายดวงชะตารายละเอียดในด้านการงาน การเงิน และความรัก
    
    Args:
        message (str): ข้อความจากผู้ใช้
        latitude (float): ละติจูดของสถานที่เกิด (ถ้าไม่ระบุจะใช้จากข้อความ)
        longitude (float): ลองจิจูดของสถานที่เกิด (ถ้าไม่ระบุจะใช้จากข้อความ)
        
    Returns:
        dict: ข้อมูลดวงชะตาพร้อมการทำนายรายละเอียด
    """
    import json
    import os
    
    parser = BirthDateParser()
    birth_info = parser.extract_birth_info(message)
    
    if not birth_info or not birth_info['date']:
        return None
    
    # ใช้พิกัดจากข้อความหรือใช้ค่าที่ส่งมา
    if latitude is None:
        latitude = birth_info.get('latitude', 13.7563)
    if longitude is None:
        longitude = birth_info.get('longitude', 100.5018)
    
    # สร้างข้อมูลดวงชะตา
    chart_info = parser.generate_birth_chart_info(birth_info['date'], birth_info['time'], latitude, longitude)
    
    if not chart_info:
        return None
    
    # เพิ่มข้อมูลสถานที่เกิด
    chart_info['birth_location_name'] = birth_info.get('location', 'กรุงเทพฯ')
    
    # โหลดข้อมูลโหราศาสตร์รายละเอียด
    zodiac_sign = chart_info['zodiac_sign']
    
    # ดึงข้อมูลราศีจาก MongoDB
    zodiac_data = get_zodiac_data_from_mongodb(zodiac_sign)
    if zodiac_data:
        chart_info['detailed_reading'] = zodiac_data
        logger.info(f"✅ Loaded zodiac data from MongoDB for {zodiac_sign}")
    else:
        logger.warning(f"⚠️ No MongoDB data found for {zodiac_sign}, falling back to JSON")
        # Fallback to JSON if MongoDB fails
        try:
            zodiac_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "zodiacData.json")
            with open(zodiac_data_path, 'r', encoding='utf-8') as f:
                json_zodiac_data = json.load(f)
            if zodiac_sign in json_zodiac_data:
                chart_info['detailed_reading'] = json_zodiac_data[zodiac_sign]
        except Exception as e:
            logger.error(f"Error loading fallback JSON data: {e}")
    
    # โหลดข้อมูลสีมงคล (ยังใช้ JSON)
    try:
        lucky_color_path = os.path.join(os.path.dirname(__file__), "..", "data", "luckyColorData.json")
        with open(lucky_color_path, 'r', encoding='utf-8') as f:
            lucky_color_data = json.load(f)
        
        # โหลดข้อมูลโชคลาภ (ยังใช้ JSON)
        omen_path = os.path.join(os.path.dirname(__file__), "..", "data", "omenData.json")
        with open(omen_path, 'r', encoding='utf-8') as f:
            omen_data = json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading color/omen data: {e}")
        lucky_color_data = {}
        omen_data = {}
    
    # ข้อมูลสีมงคล (ใช้ราศีเป็นหลัก)
    chart_info['lucky_colors'] = []
    chart_info['bad_colors'] = []
    
    # หาสีมงคลจากราศี (ใช้การประมาณการ)
    color_mapping = {
        'เมษ': 'อังคาร', 'พฤษภ': 'ศุกร์', 'เมถุน': 'พุธ', 'กรกฎ': 'จันทร์',
        'สิงห์': 'อาทิตย์', 'กันย์': 'พุธ', 'ตุล': 'ศุกร์', 'พิจิก': 'อังคาร',
        'ธนู': 'พฤหัสบดี', 'มังกร': 'เสาร์', 'กุมภ์': 'เสาร์', 'มีน': 'จันทร์'
    }
    
    ruling_planet = color_mapping.get(zodiac_sign, 'อาทิตย์')
    if ruling_planet in lucky_color_data:
        chart_info['lucky_colors'] = lucky_color_data[ruling_planet].get('luckyColors', [])
        chart_info['bad_colors'] = lucky_color_data[ruling_planet].get('badColors', [])
    
    # ข้อมูลโชคลาภ (ใช้ปีเกิด)
    try:
        birth_year = int(birth_info['date'].split('/')[2])
        thai_year = birth_year + 543
        
        # คำนวณปีนักษัตร
        animal_years = ['ชวด', 'ฉลู', 'ขาล', 'เถาะ', 'มะโรง', 'มะเส็ง', 
                       'มะเมีย', 'มะแม', 'วอก', 'ระกา', 'จอ', 'กุน']
        animal_index = (thai_year - 4) % 12
        animal_year = animal_years[animal_index]
        
        # หาข้อมูลโชคลาภจากราศีและปีนักษัตร
        ruling_planet_omens = omen_data.get(ruling_planet, {})
        if animal_year in ruling_planet_omens:
            chart_info['omen_info'] = ruling_planet_omens[animal_year]
            
    except Exception as e:
        logger.error(f"Error calculating omen info: {e}")
    
    return chart_info

def generate_birth_chart_prediction(message: str, user_id: str = "unknown") -> str:
    """
    สร้างคำทำนายดวงกำเนิดแบบละเอียดโดยใช้ RAG system (ใช้เฉพาะวันเกิด)
    
    Args:
        message (str): ข้อความจากผู้ใช้ที่มีข้อมูลวันเกิด
        user_id (str): ID ของผู้ใช้
        
    Returns:
        str: คำทำนายดวงกำเนิดแบบละเอียดจาก RAG
    """
    parser = BirthDateParser()
    birth_info = parser.extract_birth_info(message)
    
    if not birth_info or not birth_info['date']:
        return "ไม่สามารถแยกข้อมูลวันเกิดได้ กรุณาระบุวันเกิดในรูปแบบที่ชัดเจน"
    
    # สร้างข้อมูลดวงชะตา (รวมเวลาเกิดและสถานที่เกิดถ้ามี)
    chart_info = parser.generate_birth_chart_info(
        birth_info['date'], 
        birth_info.get('time'), 
        birth_info.get('latitude', 13.7563), 
        birth_info.get('longitude', 100.5018)
    )
    
    if not chart_info:
        return "ไม่สามารถสร้างข้อมูลดวงชะตาได้"
    
    # สร้างคำถามสำหรับ RAG system
    enhanced_query = create_birth_chart_query(chart_info, birth_info)
    
    # ใช้ RAG system เพื่อสร้างคำทำนาย
    try:
        from .retrieval_utils import ask_question_to_rag
        prediction = ask_question_to_rag(enhanced_query, user_id)
        
        # เพิ่มข้อมูล Ascendant ในคำตอบถ้ามี
        if 'ascendant' in chart_info and prediction:
            ascendant = chart_info['ascendant']
            ascendant_info = f"""

🌟 **ข้อมูลลัคณา (Ascendant)**
ราศีลัคณา: {ascendant['sign']} {ascendant['degree']:.1f}°
ธาตุ: {ascendant['element']}
คุณภาพ: {ascendant['quality']}

{chart_info.get('ascendant_interpretation', '')}"""
            
            # เพิ่มข้อมูลลัคณาในคำตอบ
            prediction += ascendant_info
            
            logger.info(f"✅ Added Ascendant info to response: {ascendant['sign']} {ascendant['degree']:.1f}°")
        
        return prediction
    except Exception as e:
        logger.error(f"Error in RAG system: {e}")
        return "ขออภัยครับ เกิดปัญหาในการสร้างคำทำนาย กรุณาลองใหม่อีกครั้ง"

def create_birth_chart_query(chart_info: dict, birth_info: dict) -> str:
    """
    สร้างคำถามสำหรับ RAG system เพื่อทำนายดวงกำเนิด
    
    Args:
        chart_info (dict): ข้อมูลดวงชะตา
        birth_info (dict): ข้อมูลวันเกิด เวลาเกิด และสถานที่เกิด
        
    Returns:
        str: คำถามที่เหมาะสมสำหรับ RAG system
    """
    # แปลงวันเกิดเป็นรูปแบบไทย
    day, month, year = map(int, birth_info['date'].split('/'))
    thai_year = year + 543
    
    # แปลงเดือนเป็นชื่อไทย
    thai_months = [
        '', 'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
        'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'
    ]
    
    # แปลงวันเป็นชื่อไทย
    thai_days = [
        'วันจันทร์', 'วันอังคาร', 'วันพุธ', 'วันพฤหัสบดี', 'วันศุกร์', 'วันเสาร์', 'วันอาทิตย์'
    ]
    
    # คำนวณวันในสัปดาห์
    from datetime import datetime
    birth_datetime = datetime(year, month, day)
    day_of_week = thai_days[birth_datetime.weekday()]
    
    # สร้างคำถามสำหรับ RAG
    query = f"""ทำนายดวงกำเนิดแบบละเอียดสำหรับ:
- วันเกิด: {day_of_week} {day} {thai_months[month]} พ.ศ.{thai_year}/ค.ศ.{year}"""
    
    # เพิ่มข้อมูลเวลาเกิดถ้ามี
    if birth_info.get('time'):
        query += f"\n- เวลาเกิด: {birth_info['time']}"
    
    query += f"""
- ราศีอาทิตย์: {chart_info['zodiac_sign']} ({chart_info['zodiac_element']})
- สถานที่เกิด: {birth_info.get('location', 'กรุงเทพฯ')}"""
    
    # เพิ่มข้อมูล Ascendant ถ้ามี
    if 'ascendant' in chart_info:
        ascendant = chart_info['ascendant']
        query += f"""
- ลัคณา (Ascendant): ราศี{ascendant['sign']} {ascendant['degree']:.1f}° ({ascendant['element']})"""
    
    query += """

กรุณาสร้างคำทำนายดวงกำเนิดแบบละเอียดในรูปแบบ:
1. หัวข้อ: "ทำนายดวงกำเนิด"
2. วันเกิดและราศีอาทิตย์"""
    
    # เพิ่มข้อมูลลัคณาถ้ามี
    if 'ascendant' in chart_info:
        query += """
3. ลัคณา (Ascendant) และบุคลิกภาพภายนอก"""
        section_start = 4
    else:
        section_start = 3
    
    query += f"""
{section_start}. คำทำนายลักษณะนิสัยแบบละเอียด
{section_start + 1}. ด้านการงาน
{section_start + 2}. ด้านการเงิน
{section_start + 3}. ด้านความรัก
{section_start + 4}. ด้านสุขภาพ
{section_start + 5}. สีมงคล

ใช้ข้อมูลโหราศาสตร์ตะวันตกและข้อมูลในฐานข้อมูลเพื่อสร้างคำทำนายที่แม่นยำและละเอียด"""
    
    # เพิ่มคำแนะนำสำหรับลัคณาถ้ามี
    if 'ascendant' in chart_info:
        query += """ **หากมีข้อมูลลัคณา (Ascendant) ให้รวมการตีความบุคลิกภาพภายนอกและการแสดงออกต่อสังคมด้วย**"""
    
    return query


if __name__ == "__main__":
    # ทดสอบ parser
    parser = BirthDateParser()
    parser.test_parser()
    
    
    # ทดสอบการสร้างคำทำนายดวงกำเนิด
    test_birth_chart_prediction()