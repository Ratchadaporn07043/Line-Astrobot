import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.astronomical_calculator import AstronomicalCalculator

def test_planetary_calc():
    calculator = AstronomicalCalculator()
    
    # Test case: 07/09/2003 14:30 Bangkok
    # Known: Sun in Virgo, Moon in Capricorn (approx)
    dt = datetime(2003, 9, 7, 14, 30)
    lat = 13.7563
    lon = 100.5018
    
    print(f"Testing calculation for {dt} at {lat}, {lon}")
    
    positions = calculator.calculate_planetary_positions(dt, lat, lon)
    
    if not positions:
        print("❌ Calculation failed (return None)")
        return
        
    print("✅ Calculation successful")
    print("-" * 30)
    print("Planetary Positions:")
    try:
        positions = calculator.calculate_planetary_positions(dt, lat, lon)
        
        if not positions:
            print("❌ Calculation failed (return None)")
            return
            
        print("✅ Calculation successful")
        print("-" * 30)
        print("Planetary Positions:")
        if 'planets' in positions:
            for name, data in positions['planets'].items():
                print(f"{name}: {data['sign_en']} ({data['degree']:.2f}) - House {data['house']}")
                
        print("-" * 30)
        print("Aspects:")
        aspects = calculator.calculate_aspects(positions)
        if aspects: # Assuming aspects can be empty
            for aspect in aspects:
                print(f"{aspect['p1']} {aspect['type']} {aspect['p2']} (orb {aspect['orb']})")
        
        # 🆕 Test Interpretations
        print("-" * 30)
        print("Interpretations:")
        # The get_interpretations method likely expects the full chart data, including aspects
        # Let's combine positions and aspects into a single dictionary for interpretation
        chart_data_for_interpretation = positions.copy()
        chart_data_for_interpretation['aspects'] = aspects
        
        interpretations = calculator.get_interpretations(chart_data_for_interpretation)
        if interpretations:
            for interpretation in interpretations:
                print(interpretation)
        else:
            print("No interpretations found.")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_planetary_calc()

