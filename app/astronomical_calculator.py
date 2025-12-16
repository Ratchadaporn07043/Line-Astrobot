import math
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
import logging
import os
from pymongo import MongoClient
from dotenv import load_dotenv

from .multimodel_rag import ORIGINAL_DB_NAME

# Import flatlib for accurate planetary calculations
try:
    from flatlib.datetime import Datetime as FlatlibDatetime
    from flatlib.geopos import GeoPos
    from flatlib.chart import Chart
    from flatlib import aspects, const
    FLATLIB_AVAILABLE = True
except ImportError:
    FLATLIB_AVAILABLE = False
    logging.warning("flatlib not installed. Planetary calculations will be disabled.")

logger = logging.getLogger(__name__)

class AstronomicalCalculator:
    """Class สำหรับคำนวณตำแหน่งดาวเคราะห์และ Ascendant"""
    
    def __init__(self):
        # ข้อมูลราศีและองศาเริ่มต้น (Tropical Zodiac)
        self.zodiac_signs = [
            'เมษ', 'พฤษภ', 'เมถุน', 'กรกฎ', 'สิงห์', 'กันย์',
            'ตุล', 'พิจิก', 'ธนู', 'มังกร', 'กุมภ์', 'มีน'
        ]
        
        # องศาเริ่มต้นของแต่ละราศี (0° = 0° Aries)
        self.sign_degrees = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        # ข้อมูลราศีและธาตุ
        self.sign_elements = {
            'เมษ': 'ไฟ', 'สิงห์': 'ไฟ', 'ธนู': 'ไฟ',  # Fire signs
            'พฤษภ': 'ดิน', 'กันย์': 'ดิน', 'มังกร': 'ดิน',  # Earth signs
            'เมถุน': 'ลม', 'ตุล': 'ลม', 'กุมภ์': 'ลม',  # Air signs
            'กรกฎ': 'น้ำ', 'พิจิก': 'น้ำ', 'มีน': 'น้ำ'  # Water signs
        }
        
        # ข้อมูลราศีและคุณภาพ
        self.sign_qualities = {
            'เมษ': 'Cardinal', 'กรกฎ': 'Cardinal', 'ตุล': 'Cardinal', 'มังกร': 'Cardinal',  # Cardinal
            'พฤษภ': 'Fixed', 'สิงห์': 'Fixed', 'พิจิก': 'Fixed', 'กุมภ์': 'Fixed',  # Fixed
            'เมถุน': 'Mutable', 'กันย์': 'Mutable', 'ธนู': 'Mutable', 'มีน': 'Mutable'  # Mutable
        }

        # ตั้งค่า MongoDB (ถ้าถูกกำหนดไว้)
        load_dotenv()
        self._mongo_client: Optional[MongoClient] = None
        self._mongo_db = None
        try:
            mongo_uri = os.getenv("MONGO_URL")
            if mongo_uri and mongo_uri != "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
                self._mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
                self._mongo_db = self._mongo_client[ORIGINAL_DB_NAME]
        except Exception as conn_err:
            logger.warning(f"ไม่สามารถเริ่มต้น MongoDB client ได้: {conn_err}")

    def _get_collection(self, name: str):
        """คืนค่า collection จาก MongoDB หากพร้อมใช้งาน มิฉะนั้นคืนค่า None"""
        try:
            return self._mongo_db[name] if self._mongo_db is not None else None
        except Exception:
            return None

    def calculate_ascendant(self, birth_datetime: datetime, latitude: float, longitude: float) -> Dict:
        """
        คำนวณ Ascendant (ราศีประจำลัคนา) จากเวลาเกิดและสถานที่เกิด
        
        Args:
            birth_datetime (datetime): เวลาเกิด
            latitude (float): ละติจูดของสถานที่เกิด
            longitude (float): ลองจิจูดของสถานที่เกิด
            
        Returns:
            dict: ข้อมูล Ascendant {'sign': 'ชื่อราศี', 'degree': float, 'element': 'ธาตุ', 'quality': 'คุณภาพ'}
        """
        try:
            # คำนวณ Local Sidereal Time (LST)
            lst = self._calculate_lst(birth_datetime, longitude)
            
            # คำนวณ Ascendant degree
            ascendant_degree = self._calculate_ascendant_degree(lst, latitude)
            
            # หาราศีและองศาในราศี
            sign_index = int(ascendant_degree // 30)
            degree_in_sign = ascendant_degree % 30
            
            # ตรวจสอบขอบเขต
            if sign_index >= 12:
                sign_index = 0
            
            ascendant_sign = self.zodiac_signs[sign_index]
            
            return {
                'sign': ascendant_sign,
                'degree': round(degree_in_sign, 2),
                'element': self.sign_elements.get(ascendant_sign, ''),
                'quality': self.sign_qualities.get(ascendant_sign, ''),
                'full_degree': round(ascendant_degree, 2)
            }
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Ascendant: {e}")
            return None

    def _calculate_lst(self, birth_datetime: datetime, longitude: float) -> float:
        """
        คำนวณ Local Sidereal Time (LST)
        
        Args:
            birth_datetime (datetime): เวลาเกิด
            longitude (float): ลองจิจูด
            
        Returns:
            float: Local Sidereal Time ในหน่วยองศา
        """
        # แปลงเป็น UTC
        if birth_datetime.tzinfo is None:
            birth_datetime = birth_datetime.replace(tzinfo=timezone.utc)
        
        # คำนวณ Julian Day
        jd = self._datetime_to_julian_day(birth_datetime)
        
        # คำนวณ Greenwich Sidereal Time (GST)
        gst = self._calculate_gst(jd)
        
        # คำนวณ Local Sidereal Time (LST)
        lst = gst + longitude
        
        # ปรับให้อยู่ในช่วง 0-360 องศา
        lst = lst % 360
        if lst < 0:
            lst += 360
            
        return lst

    def _datetime_to_julian_day(self, dt: datetime) -> float:
        """
        แปลง datetime เป็น Julian Day
        
        Args:
            dt (datetime): เวลา
            
        Returns:
            float: Julian Day
        """
        # แปลงเป็น UTC
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        
        # คำนวณ Julian Day
        if month <= 2:
            year -= 1
            month += 12
        
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
        
        return jd

    def _calculate_gst(self, jd: float) -> float:
        """
        คำนวณ Greenwich Sidereal Time (GST)
        
        Args:
            jd (float): Julian Day
            
        Returns:
            float: GST ในหน่วยองศา
        """
        # คำนวณ T (จำนวนศตวรรษ Julian ตั้งแต่ J2000.0)
        t = (jd - 2451545.0) / 36525.0
        
        # คำนวณ GST
        gst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t * t * t / 38710000.0
        
        # ปรับให้อยู่ในช่วง 0-360 องศา
        gst = gst % 360
        if gst < 0:
            gst += 360
            
        return gst

    def _calculate_ascendant_degree(self, lst: float, latitude: float) -> float:
        """
        คำนวณองศา Ascendant จาก LST และ latitude
        
        Args:
            lst (float): Local Sidereal Time ในหน่วยองศา
            latitude (float): ละติจูด
            
        Returns:
            float: องศา Ascendant
        """
        # แปลงเป็นเรเดียน
        lst_rad = math.radians(lst)
        lat_rad = math.radians(latitude)
        
        # คำนวณ Ascendant degree
        # ใช้สูตร: tan(ASC) = cos(LST) / (sin(LST) * cos(lat) + tan(lat) * sin(obliquity))
        # สำหรับความง่าย ใช้การประมาณการแบบง่าย
        
        # คำนวณ Obliquity of the Ecliptic (ประมาณ 23.44°)
        obliquity = 23.44
        obliquity_rad = math.radians(obliquity)
        
        # คำนวณ Ascendant
        numerator = math.cos(lst_rad)
        denominator = math.sin(lst_rad) * math.cos(lat_rad) + math.tan(lat_rad) * math.sin(obliquity_rad)
        
        if abs(denominator) < 1e-10:  # หลีกเลี่ยงการหารด้วยศูนย์
            ascendant_rad = 0
        else:
            ascendant_rad = math.atan2(numerator, denominator)
        
        # แปลงกลับเป็นองศา
        ascendant_degree = math.degrees(ascendant_rad)
        
        # ปรับให้อยู่ในช่วง 0-360 องศา
        if ascendant_degree < 0:
            ascendant_degree += 360
            
        return ascendant_degree

    def get_ascendant_interpretation(self, ascendant_data: Dict) -> str:
        """
        สร้างการตีความ Ascendant
        
        Args:
            ascendant_data (dict): ข้อมูล Ascendant
            
        Returns:
            str: การตีความ Ascendant
        """
        if not ascendant_data:
            return "ไม่สามารถคำนวณ Ascendant ได้"
        
        sign = ascendant_data['sign']
        degree = ascendant_data['degree']
        element = ascendant_data['element']
        quality = ascendant_data['quality']

        # ดึงการตีความจาก MongoDB
        interpretation_text = None
        try:
            collection = self._get_collection('ascendant_interpretations')
            if collection is not None:
                doc = collection.find_one({"sign": sign})
                if doc:
                    if isinstance(doc.get('interpretation'), str) and doc['interpretation'].strip():
                        interpretation_text = doc['interpretation'].strip()
                    elif isinstance(doc.get('text'), str) and doc['text'].strip():
                        interpretation_text = doc['text'].strip()
        except Exception as db_err:
            logger.warning(f"ไม่สามารถดึงการตีความ Ascendant จากฐานข้อมูลได้: {db_err}")

        if not interpretation_text:
            interpretation_text = "การตีความลัคณาจะดึงจากฐานข้อมูลเมื่อมีการตั้งค่า"

        degree_info = f" (องศา {degree:.1f}° ในราศี{sign})"
        element_quality_info = f" เป็นราศีธาตุ{element} และมีคุณภาพ{quality}"

        return interpretation_text + degree_info + element_quality_info

    def calculate_house_cusps(self, birth_datetime: datetime, latitude: float, longitude: float) -> Dict:
        """
        คำนวณตำแหน่งบ้านทั้ง 12 บ้าน (House Cusps)
        
        Args:
            birth_datetime (datetime): เวลาเกิด
            latitude (float): ละติจูด
            longitude (float): ลองจิจูด
            
        Returns:
            dict: ข้อมูลบ้านทั้ง 12 บ้าน
        """
        try:
            # คำนวณ Ascendant
            ascendant_data = self.calculate_ascendant(birth_datetime, latitude, longitude)
            if not ascendant_data:
                return None
            
            ascendant_degree = ascendant_data['full_degree']
            
            # คำนวณบ้านทั้ง 12 บ้าน (ใช้ระบบ Equal House)
            houses = {}
            for i in range(1, 13):
                house_degree = (ascendant_degree + (i - 1) * 30) % 360
                sign_index = int(house_degree // 30)
                degree_in_sign = house_degree % 30
                
                if sign_index >= 12:
                    sign_index = 0
                
                sign_name = self.zodiac_signs[sign_index]
                
                houses[f'house_{i}'] = {
                    'sign': sign_name,
                    'degree': round(degree_in_sign, 2),
                    'full_degree': round(house_degree, 2),
                    'element': self.sign_elements.get(sign_name, ''),
                    'quality': self.sign_qualities.get(sign_name, '')
                }
            
            return houses
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณ house cusps: {e}")
            return None

    def calculate_planetary_positions(self, birth_datetime: datetime, latitude: float, longitude: float) -> Dict:
        """
        คำนวณตำแหน่งดาวเคราะห์ (Planetary Positions)
        
        Args:
            birth_datetime (datetime): เวลาเกิด
            latitude (float): ละติจูด
            longitude (float): ลองจิจูด
            
        Returns:
            dict: ข้อมูลตำแหน่งดาวเคราะห์
        """
        if not FLATLIB_AVAILABLE:
            return None
            
        try:
            # แปลงเวลาเป็น UTC
            if birth_datetime.tzinfo is None:
                # ถ้าไม่มี timezone ให้สมมติว่าเป็น Local Time (BKK UTC+7)
                # แต่ flatlib ต้องการ UTC หรือ string format ที่ชัดเจน
                # วิธีที่ง่ายที่สุดคือแปลงเป็น string YYYY/MM/DD HH:MM และระบุ Offset
                # สำหรับประเทศไทย UTC+7
                pass
            
            # สร้าง Flatlib Datetime
            # flatlib รับ date เป็น 'YYYY/MM/DD' และ time เป็น 'HH:MM' และ utcoffset เป็น signed float (e.g. +7)
            date_str = birth_datetime.strftime('%Y/%m/%d')
            time_str = birth_datetime.strftime('%H:%M')
            
            # สร้าง GeoPos
            pos = GeoPos(latitude, longitude)
            
            # สร้าง Datetime object (UTC+7 สำหรับประเทศไทย)
            # หมายเหตุ: ใน production ควรปรับตาม timezone จริงของผู้ใช้ แต่ตอนนี้ใช้ +7 ไปก่อน
            date = FlatlibDatetime(date_str, time_str, '+07:00')
            
            # คำนวณ Chart
            chart = Chart(date, pos, IDs=const.LIST_OBJECTS)
            
            planets = {}
            thai_names = {
                'Sun': 'อาทิตย์', 'Moon': 'จันทร์', 'Mercury': 'พุธ', 'Venus': 'ศุกร์',
                'Mars': 'อังคาร', 'Jupiter': 'พฤหัสบดี', 'Saturn': 'เสาร์',
                'Uranus': 'มฤตยู', 'Neptune': 'เนปจูน', 'Pluto': 'พลูโต',
                'Chiron': 'ไครอน', 'North Node': 'ราหู', 'South Node': 'เกตุ'
            }
            
            thai_zodiacs = {
                'Aries': 'เมษ', 'Taurus': 'พฤษภ', 'Gemini': 'เมถุน', 'Cancer': 'กรกฎ',
                'Leo': 'สิงห์', 'Virgo': 'กันย์', 'Libra': 'ตุล', 'Scorpio': 'พิจิก',
                'Sagittarius': 'ธนู', 'Capricorn': 'มังกร', 'Aquarius': 'กุมภ์', 'Pisces': 'มีน'
            }

            for obj in const.LIST_OBJECTS:
                planet = chart.get(obj)
                name = getattr(planet, 'id', str(obj)) # ป้องกัน attribute error
                sign = getattr(planet, 'sign', '')
                lon = getattr(planet, 'lon', 0.0) # Absolute longitude
                signlon = getattr(planet, 'signlon', 0.0) # Degree in sign
                
                # หา House (ต้องคำนวณ house แยก หรือดูจาก chart houses ถ้า flatlib map ให้)
                # Flatlib's chart.get(obj) does not strictly return house. 
                # We can check which house it falls into based on chart.houses
                house_num = -1
                for h_obj in const.LIST_HOUSES:
                    house = chart.get(h_obj)
                    if house.hasObject(planet):
                        house_num = house.id.replace('House', '')
                        break
                
                planets[name] = {
                    'name_en': name,
                    'name_th': thai_names.get(name, name),
                    'sign_en': sign,
                    'sign_th': thai_zodiacs.get(sign, sign),
                    'degree': round(signlon, 2),
                    'absolute_degree': round(lon, 2),
                    'house': house_num,
                    'retrograde': planet.isRetrograde()  # ตรวจสอบการเดินถอยหลัง
                }
                
            return {
                'planets': planets,
                'chart_object': chart  # เก็บ object ไว้คำนวณ aspect ต่อ
            }
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณตำแหน่งดาวเคราะห์: {e}")
            return None

    def calculate_aspects(self, chart_data: Dict) -> list:
        """
        คำนวณมุมสัมพันธ์ (Aspects)
        
        Args:
            chart_data (dict): ข้อมูล Chart ที่ได้จาก calculate_planetary_positions
            
        Returns:
            list: รายการมุมสัมพันธ์
        """
        if not FLATLIB_AVAILABLE or not chart_data or 'chart_object' not in chart_data:
            return []
            
        try:
            chart = chart_data['chart_object']
            aspect_list = []
            
            # มุมสัมพันธ์หลัก
            major_aspects = [const.CONJUNCTION, const.SEXTILE, const.SQUARE, const.TRINE, const.OPPOSITION]
            thai_aspects = {
                const.CONJUNCTION: 'กุม',
                const.SEXTILE: 'โยค', 
                const.SQUARE: 'ฉาก',
                const.TRINE: 'ตรีโกณ',
                const.OPPOSITION: 'เล็ง'
            }
            
            thai_names = {
                'Sun': 'อาทิตย์', 'Moon': 'จันทร์', 'Mercury': 'พุธ', 'Venus': 'ศุกร์',
                'Mars': 'อังคาร', 'Jupiter': 'พฤหัสบดี', 'Saturn': 'เสาร์',
                'Uranus': 'มฤตยู', 'Neptune': 'เนปจูน', 'Pluto': 'พลูโต',
                'North Node': 'ราหู', 'South Node': 'เกตุ'
            }
            
            # วนลูปหา aspect ของดาวเคราะห์แต่ละคู่
            objects = [obj for obj in const.LIST_OBJECTS if obj in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']]
            
            for i, p1_name in enumerate(objects):
                for p2_name in objects[i+1:]:
                    p1 = chart.get(p1_name)
                    p2 = chart.get(p2_name)
                    
                    # คำนวณ aspect exactness
                    # ใช้ aspects.getAspect แทน chart.getAspect
                    aspect = aspects.getAspect(p1, p2, major_aspects)
                    
                    if aspect.exists():
                        aspect_list.append({
                            'p1': p1_name,
                            'p1_th': thai_names.get(p1_name, p1_name),
                            'p2': p2_name,
                            'p2_th': thai_names.get(p2_name, p2_name),
                            'type': aspect.type,
                            'type_th': thai_aspects.get(aspect.type, aspect.type),
                            'orb': round(aspect.orb, 2)
                        })
            
            return aspect_list
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณมุมสัมพันธ์: {e}")
            return []

    def get_house_interpretation(self, house_number: int, house_data: Dict) -> str:
        """
        สร้างการตีความบ้าน
        
        Args:
            house_number (int): หมายเลขบ้าน (1-12)
            house_data (dict): ข้อมูลบ้าน
            
        Returns:
            str: การตีความบ้าน
        """
        if not house_data:
            return f"ไม่สามารถคำนวณบ้านที่ {house_number} ได้"
        
        sign = house_data['sign']
        degree = house_data['degree']

        # พยายามดึงคำอธิบายบ้านจาก MongoDB
        meaning = None
        try:
            collection = self._get_collection('house_interpretations') or self._get_collection('house_meanings')
            if collection is not None:
                # รองรับทั้งฟิลด์ house_number และ number
                doc = collection.find_one({"house_number": house_number}) or collection.find_one({"number": house_number})
                if doc:
                    # รองรับฟิลด์ meaning หรือ description หรือ text
                    for key in ["meaning", "description", "text"]:
                        if isinstance(doc.get(key), str) and doc[key].strip():
                            meaning = doc[key].strip()
                            break
        except Exception as db_err:
            logger.warning(f"ไม่สามารถดึงการตีความบ้านจากฐานข้อมูลได้: {db_err}")

        if not meaning:
            meaning = "คำอธิบายบ้านจะดึงจากฐานข้อมูลเมื่อมีการตั้งค่า"

        return f"บ้านที่ {house_number} ({meaning}): ราศี{sign} องศา {degree:.1f}°"

    def test_calculator(self):
        """ทดสอบเครื่องคำนวณดาราศาสตร์"""
        print("🧪 ทดสอบ Astronomical Calculator")
        print("=" * 50)
        
        # ทดสอบข้อมูลตัวอย่าง
        test_cases = [
            {
                'name': 'กรุงเทพฯ',
                'datetime': datetime(1990, 3, 15, 14, 30),  # 15 มีนาคม 1990 เวลา 14:30
                'latitude': 13.7563,
                'longitude': 100.5018
            },
            {
                'name': 'เชียงใหม่',
                'datetime': datetime(1985, 7, 20, 8, 15),  # 20 กรกฎาคม 1985 เวลา 8:15
                'latitude': 18.7883,
                'longitude': 98.9853
            },
            {
                'name': 'ภูเก็ต',
                'datetime': datetime(1995, 12, 10, 22, 45),  # 10 ธันวาคม 1995 เวลา 22:45
                'latitude': 7.8804,
                'longitude': 98.3923
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{i}. ทดสอบ {test['name']}")
            print(f"   เวลาเกิด: {test['datetime']}")
            print(f"   พิกัด: {test['latitude']:.4f}°N, {test['longitude']:.4f}°E")
            
            # คำนวณ Ascendant
            ascendant = self.calculate_ascendant(
                test['datetime'], 
                test['latitude'], 
                test['longitude']
            )
            
            if ascendant:
                print(f"   🌟 Ascendant: ราศี{ascendant['sign']} {ascendant['degree']:.1f}°")
                print(f"   🔥 ธาตุ: {ascendant['element']}")
                print(f"   ⚡ คุณภาพ: {ascendant['quality']}")
                
                # แสดงการตีความ
                interpretation = self.get_ascendant_interpretation(ascendant)
                print(f"   📝 การตีความ: {interpretation}")
                
                # คำนวณบ้านทั้ง 12 บ้าน
                houses = self.calculate_house_cusps(
                    test['datetime'], 
                    test['latitude'], 
                    test['longitude']
                )
                
                if houses:
                    print(f"   🏠 บ้านทั้ง 12 บ้าน:")
                    for house_num in range(1, 13):
                        house_data = houses[f'house_{house_num}']
                        house_interpretation = self.get_house_interpretation(house_num, house_data)
                        print(f"      {house_interpretation}")
            else:
                print("   ❌ ไม่สามารถคำนวณได้")
            
            print("-" * 50)

if __name__ == "__main__":
    # ทดสอบเครื่องคำนวณ
    calculator = AstronomicalCalculator()
    calculator.test_calculator()
