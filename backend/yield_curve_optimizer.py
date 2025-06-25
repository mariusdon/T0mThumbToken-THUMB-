import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

class YieldCurveOptimizer:
    def __init__(self):
        self.maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        self.fred_codes = {
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
    def get_current_yields(self) -> Dict[str, float]:
        """Fetch current Treasury yield data from FRED"""
        end = datetime.now()
        start = end - timedelta(days=7)  # Get last week to ensure we have recent data
        
        yields = {}
        for mat, code in self.fred_codes.items():
            try:
                df = pdr.DataReader(code, 'fred', start, end)
                # Get most recent non-null value
                latest = df[df.iloc[:, 0].notna()].iloc[-1, 0]
                yields[mat] = float(latest)
            except Exception as e:
                print(f"Error fetching {mat}: {e}")
                yields[mat] = None
                
        return yields

    def calculate_allocation(self, yields: Dict[str, float]) -> Dict[str, int]:
        """Calculate allocation based on current yield curve shape"""
        # Remove any None values
        valid_yields = {k: v for k, v in yields.items() if v is not None}
        if not valid_yields:
            return {f'Thumb{mat}': 1428 for mat in self.maturities}  # Equal distribution if no data
        
        # Calculate yield curve metrics
        max_yield = max(valid_yields.values())
        min_yield = min(valid_yields.values())
        yield_range = max_yield - min_yield if max_yield > min_yield else 1
        
        # Calculate weights based on relative yields
        weights = {}
        total_weight = 0
        for mat in self.maturities:
            if mat in valid_yields:
                # Higher yields get higher weights
                weight = 1000 + int(3000 * (valid_yields[mat] - min_yield) / yield_range)
                weights[f'Thumb{mat}'] = weight
                total_weight += weight
            else:
                weights[f'Thumb{mat}'] = 1000
                total_weight += 1000
        
        # Normalize to 10000 basis points
        allocation = {k: int(v * 10000 / total_weight) for k, v in weights.items()}
        
        # Ensure we sum to exactly 10000
        total = sum(allocation.values())
        if total != 10000:
            # Add/subtract the difference from the largest allocation
            diff = 10000 - total
            max_key = max(allocation, key=allocation.get)
            allocation[max_key] += diff
        
        return allocation

    def get_optimal_allocation(self) -> Dict:
        """Get optimal allocation weights in basis points"""
        # Get current yields
        current_yields = self.get_current_yields()
        
        # Calculate allocation
        allocation = self.calculate_allocation(current_yields)
        
        # Add metadata
        result = {
            'allocation': allocation,
            'timestamp': datetime.now().isoformat(),
            'current_yields': current_yields,
            'strategy': 'Yield-weighted allocation strategy'
        }
        
        return result 