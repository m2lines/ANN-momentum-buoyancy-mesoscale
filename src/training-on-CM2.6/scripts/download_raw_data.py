import sys
sys.path.append('../')
import cftime
import os
import time

from helpers.cm26 import DatasetCM26

# Read data in cloud
ds = DatasetCM26(source='cmip6-3d')
PATH = os.path.expandvars(os.environ.get('CM26_RAWDATA', '/scratch/$USER/CM26_datasets/ocean3d/rawdata'))

ds.param.to_netcdf(f'{PATH}/param.nc')

# Precompute some features
data = ds.data
data['rho'] = ds.state.rho()
data = data.drop_vars(['temp', 'salt'])
data = data.astype('float32')
data['rho'] = data['rho'].transpose('time', 'zl', 'yh', 'xh')

# Select time range for train/validate/test split
train_selector = lambda x: x.sel(time=[cftime.DatetimeJulian(year,month,15) for year in range(181,189) for month in range(1,13)], method='nearest')
validate_selector = lambda x: x.sel(time=[cftime.DatetimeJulian(year,month,15) for year in range(194,195) for month in range(1,13)], method='nearest')
test_selector = lambda x: x.sel(time=[cftime.DatetimeJulian(year,month,15) for year in range(199,201) for month in range(1,13)], method='nearest')

# Save selected snapshots on disk

## train dataset 
train_data = train_selector(data)
steps = len(train_data.time)
t_s = time.time()
for step in range(steps):
    t_e = time.time()
    train_data.isel(time=step).to_netcdf(f'{PATH}/train-{step}.nc')
    t = time.time()
    print(f'Train: [{step+1}/{steps}]'+', Step time/ETA: [%d/%d]' % (t-t_e, (t-t_s)*(steps/(step+1)-1)))

## validate dataset 
validate_data = validate_selector(data)
steps = len(validate_data.time)
t_s = time.time()
for step in range(steps):
    t_e = time.time()
    validate_data.isel(time=step).to_netcdf(f'{PATH}/validate-{step}.nc')
    t = time.time()
    print(f'Validate: [{step+1}/{steps}]'+', Step time/ETA: [%d/%d]' % (t-t_e, (t-t_s)*(steps/(step+1)-1)))

## test dataset
test_data = test_selector(data)
steps = len(test_data.time)
t_s = time.time()
for step in range(steps):
    t_e = time.time()
    test_data.isel(time=step).to_netcdf(f'{PATH}/test-{step}.nc')
    t = time.time()
    print(f'Test: [{step+1}/{steps}]'+', Step time/ETA: [%d/%d]' % (t-t_e, (t-t_s)*(steps/(step+1)-1)))