"""
NASA Exoplanet Data Acquisition Module
Fetches and preprocesses data from Kepler and TESS missions
"""

import os
import numpy as np
import pandas as pd
import requests
from astroquery.mast import Observations
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import lightkurve as lk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDataAcquisition:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f'{data_dir}/kepler', exist_ok=True)
        os.makedirs(f'{data_dir}/tess', exist_ok=True)
        os.makedirs(f'{data_dir}/processed', exist_ok=True)
        
    def get_confirmed_exoplanets(self, limit=1000):
        """Fetch confirmed exoplanet data from NASA Exoplanet Archive"""
        print("Fetching confirmed exoplanet data...")
        
        try:
            # Get confirmed exoplanets with Kepler/TESS data
            exoplanets = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name,hostname,pl_orbper,pl_rade,pl_masse,st_teff,st_rad,st_mass,disc_facility",
                where="disc_facility like '%Kepler%' or disc_facility like '%TESS%'",
                order="pl_name"
            )
            
            # Limit results for faster processing
            if len(exoplanets) > limit:
                exoplanets = exoplanets[:limit]
                
            df = exoplanets.to_pandas()
            df.to_csv(f'{self.data_dir}/confirmed_exoplanets.csv', index=False)
            print(f"Downloaded {len(df)} confirmed exoplanets")
            return df
            
        except Exception as e:
            print(f"Error fetching exoplanet data: {e}")
            return self._create_sample_exoplanet_data()
    
    def _create_sample_exoplanet_data(self):
        """Create sample exoplanet data if API fails"""
        print("Creating sample exoplanet data...")
        sample_data = {
            'pl_name': ['Kepler-452b', 'TRAPPIST-1b', 'TOI-715b', 'K2-18b'],
            'hostname': ['Kepler-452', 'TRAPPIST-1', 'TOI-715', 'K2-18'],
            'pl_orbper': [384.8, 1.51, 19.3, 32.9],
            'pl_rade': [1.63, 1.09, 1.55, 2.3],
            'st_teff': [5757, 2559, 4600, 3457],
            'disc_facility': ['Kepler', 'TRAPPIST', 'TESS', 'Kepler']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(f'{self.data_dir}/confirmed_exoplanets.csv', index=False)
        return df
    
    def download_kepler_light_curves(self, target_list=None, max_targets=50):
        """Download Kepler light curves for known exoplanet hosts"""
        print("Downloading Kepler light curves...")
        
        if target_list is None:
            # Use some well-known Kepler targets
            target_list = [
                'Kepler-452', 'Kepler-186', 'Kepler-442', 'Kepler-438', 'Kepler-296',
                'Kepler-62', 'Kepler-283', 'Kepler-296', 'Kepler-440', 'Kepler-442'
            ]
        
        light_curves = []
        
        for i, target in enumerate(tqdm(target_list[:max_targets], desc="Downloading Kepler data")):
            try:
                # Search for light curves
                search_result = lk.search_lightcurve(target, mission='Kepler')
                
                if len(search_result) > 0:
                    # Download first available quarter
                    lc = search_result[0].download()
                    if lc is not None:
                        # Normalize and clean the light curve
                        lc = lc.normalize().remove_outliers()
                        
                        # Save processed light curve
                        filename = f'{self.data_dir}/kepler/{target.replace(" ", "_")}_lc.fits'
                        lc.to_fits(filename, overwrite=True)
                        
                        light_curves.append({
                            'target': target,
                            'filename': filename,
                            'length': len(lc.flux),
                            'has_exoplanet': True  # These are known exoplanet hosts
                        })
                        
            except Exception as e:
                print(f"Error downloading {target}: {e}")
                continue
                
        print(f"Downloaded {len(light_curves)} Kepler light curves")
        return light_curves
    
    def download_tess_light_curves(self, target_list=None, max_targets=30):
        """Download TESS light curves"""
        print("Downloading TESS light curves...")
        
        if target_list is None:
            # Use some TESS targets
            target_list = [
                'TOI-715', 'TOI-849', 'TOI-178', 'TOI-270', 'TOI-421',
                'TOI-561', 'TOI-674', 'TOI-700', 'TOI-1338', 'TOI-1452'
            ]
        
        light_curves = []
        
        for target in tqdm(target_list[:max_targets], desc="Downloading TESS data"):
            try:
                search_result = lk.search_lightcurve(target, mission='TESS')
                
                if len(search_result) > 0:
                    lc = search_result[0].download()
                    if lc is not None:
                        lc = lc.normalize().remove_outliers()
                        
                        filename = f'{self.data_dir}/tess/{target.replace(" ", "_")}_lc.fits'
                        lc.to_fits(filename, overwrite=True)
                        
                        light_curves.append({
                            'target': target,
                            'filename': filename,
                            'length': len(lc.flux),
                            'has_exoplanet': True
                        })
                        
            except Exception as e:
                print(f"Error downloading {target}: {e}")
                continue
                
        print(f"Downloaded {len(light_curves)} TESS light curves")
        return light_curves
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic light curves for training"""
        print("Generating synthetic light curves...")
        
        synthetic_data = []
        
        for i in tqdm(range(n_samples), desc="Generating synthetic data"):
            # Create time array
            time = np.linspace(0, 90, 3000)  # 90 days, ~3000 points
            
            # Base stellar flux with noise
            flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
            
            # Add stellar variability
            flux += 0.01 * np.sin(2 * np.pi * time / 25.0)  # 25-day rotation
            
            # Decide if this light curve has a transit
            has_transit = np.random.choice([True, False], p=[0.3, 0.7])
            
            if has_transit:
                # Add transit signal
                period = np.random.uniform(1, 50)  # Orbital period in days
                depth = np.random.uniform(0.001, 0.02)  # Transit depth
                duration = np.random.uniform(0.05, 0.3)  # Transit duration
                
                # Calculate transit times
                transit_times = np.arange(0, 90, period)
                
                for t_transit in transit_times:
                    if t_transit < 90:
                        # Create transit shape (simplified)
                        transit_mask = np.abs(time - t_transit) < duration/2
                        flux[transit_mask] -= depth
            
            # Save synthetic light curve
            filename = f'{self.data_dir}/processed/synthetic_{i:04d}.npy'
            np.save(filename, {'time': time, 'flux': flux, 'has_exoplanet': has_transit})
            
            synthetic_data.append({
                'filename': filename,
                'has_exoplanet': has_transit,
                'length': len(flux),
                'synthetic': True
            })
        
        print(f"Generated {len(synthetic_data)} synthetic light curves")
        return synthetic_data
    
    def create_training_dataset(self):
        """Create a comprehensive training dataset"""
        print("Creating comprehensive training dataset...")
        
        # Get confirmed exoplanet data
        exoplanets_df = self.get_confirmed_exoplanets(limit=100)
        
        # Download real light curves (smaller sample for speed)
        kepler_lcs = self.download_kepler_light_curves(max_targets=20)
        tess_lcs = self.download_tess_light_curves(max_targets=15)
        
        # Generate synthetic data
        synthetic_lcs = self.generate_synthetic_data(n_samples=500)
        
        # Combine all data
        all_data = kepler_lcs + tess_lcs + synthetic_lcs
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(all_data)
        metadata_df.to_csv(f'{self.data_dir}/training_metadata.csv', index=False)
        
        print(f"Training dataset created with {len(all_data)} light curves")
        print(f"- Kepler: {len(kepler_lcs)}")
        print(f"- TESS: {len(tess_lcs)}")
        print(f"- Synthetic: {len(synthetic_lcs)}")
        
        return metadata_df

if __name__ == "__main__":
    # Initialize data acquisition
    data_acq = ExoplanetDataAcquisition()
    
    # Create training dataset
    dataset = data_acq.create_training_dataset()
    
    print("Data acquisition complete!")
    print(f"Dataset summary:\n{dataset.groupby('has_exoplanet').size()}")
