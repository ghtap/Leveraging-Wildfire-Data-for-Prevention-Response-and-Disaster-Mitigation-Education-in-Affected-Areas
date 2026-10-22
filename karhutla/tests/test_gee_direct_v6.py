"""Tes fitur direct pada fetch_climate v6 tanpa akun GEE / jaringan."""
import sys
import types
from pathlib import Path
import pandas as pd
import pytest
from src.data import fetch_climate as fc

@pytest.fixture
def world(tmp_path, monkeypatch):
    (tmp_path / 'data' / 'interim').mkdir(parents=True)
    cfg = {'paths': {'interim': 'data/interim'},
           'climate': {'gee_dir': 'data/raw/gee', 'gee_project': 'project-test',
                       'chirps_asset': 'UCSB-CHC/CHIRPS/V3/DAILY_SAT'},
           'dates': {'start': '2025-08-01', 'end': '2025-08-03'}}
    pd.DataFrame({'cell_id':[10,11], 'province':['Riau','Riau'],
                  'lon':[101.0,101.1], 'lat':[0.5,0.6]}).to_csv(
        tmp_path/'data/interim/grid.csv', index=False)
    monkeypatch.setattr(fc, 'rpath', lambda rel: tmp_path / rel)
    ee = types.ModuleType('ee')
    ee.Initialize = lambda project: None
    monkeypatch.setitem(sys.modules,'ee',ee)
    # Test ortam here doesn't have pyarrow; Windows user's environment does.
    monkeypatch.setattr(pd.DataFrame, 'to_parquet',
                        lambda self, path, **kwargs: self.to_pickle(path))
    monkeypatch.setattr(pd, 'read_parquet',
                        lambda path, **kwargs: pd.read_pickle(path))
    return cfg,tmp_path


def test_direct_skips_old_and_writes_raw_plus_parquet(world,monkeypatch):
    cfg,root=world
    fc.buat_rencana(cfg,source='both',start='2025-08-01',end='2025-08-03',
                    province='Riau',cells_per_batch=2,days_per_batch=2)
    calls=[]
    def fake_fetch(ee,cfg,rec,sub):
        calls.append(rec.source)
        rows=[]
        for date in ['2025-08-01','2025-08-02']:
            for i in [10,11]:
                r={'cell_id':i,'date':date}
                if rec.source=='chirps':r['precipitation']=0. if i==10 else 3.
                else:r.update({'temperature_2m':300.,'dewpoint_temperature_2m':294.,
                               'volumetric_soil_water_layer_1':0.2,
                               'u_component_of_wind_10m':1.,'v_component_of_wind_10m':2.})
                rows.append(r)
        return pd.DataFrame(rows)
    monkeypatch.setattr(fc,'_fetch_direct_dataframe',fake_fetch)
    fc.ambil_langsung(cfg,max_batches=2)
    manifest=fc._read_manifest(cfg)
    assert set(manifest.status)=={'INGESTED'}
    assert calls==['chirps','era5']
    for _,rec in manifest.iterrows():
        assert (root/'data/raw/gee/batches'/(rec['name']+'.csv')).exists()
        path=root/'data/interim/climate_parts'/rec.source/(rec['name']+'.parquet')
        assert len(pd.read_parquet(path))==4
    fc.ambil_langsung(cfg,max_batches=2)
    assert calls==['chirps','era5']
    # Parquet writer/read mocked to pickle for environment without pyarrow.
    # Verify two validated Parquet parts were produced, without real engine.
    assert len(calls)==2


def test_direct_prefers_missing_pair(world,monkeypatch):
    cfg,root=world
    fc.buat_rencana(cfg,source='both',start='2025-08-01',end='2025-08-03',
                    province='Riau',cells_per_batch=2,days_per_batch=2)
    m=fc._read_manifest(cfg)
    m.loc[m.source.eq('chirps'),'status']='INGESTED'
    fc._save_manifest(cfg,m)
    choices=fc._direct_candidates(fc._read_manifest(cfg))
    assert len(choices)==1 and choices.iloc[0].source=='era5'


def test_direct_skips_unmanageable_rows(world,monkeypatch,capsys):
    cfg,_=world
    fc.buat_rencana(cfg,source='chirps',start='2025-08-01',end='2025-08-03',
                    province='Riau',cells_per_batch=2,days_per_batch=2)
    m=fc._read_manifest(cfg)
    m.loc[:,'expected_rows']=8000
    fc._save_manifest(cfg,m)
    fc.ambil_langsung(cfg,max_batches=1)
    assert 'SKIP' in capsys.readouterr().out
    assert fc._read_manifest(cfg).iloc[0].status=='PLANNED'
