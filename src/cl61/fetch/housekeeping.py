import requests
import xarray as xr
import io
import pandas as pd
import time

def fetch_housekeeping(fetch_func, site, start_date, end_date, save_path):
    url = 'https://cloudnet.fmi.fi/api/raw-files'
    pr = pd.period_range(start=start_date,end=end_date, freq='D') 

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        df_save = {'diag': pd.DataFrame({}),
                   'monitoring': pd.DataFrame({}),
                   'status': pd.DataFrame({})}
        params = {
            'dateFrom': idate,
            'dateTo': idate,
            'site': site,
            'instrument': 'cl61d'
        }
        metadata = requests.get(url, params).json()
        if not metadata:
            continue
        
        for row in metadata:
            if 'live' in row['filename']:
                if int(row['size']) < 100000:
                    continue
                while True:
                    try:
                        print(row['filename'])
                        bad_file=False
                        df_save = fetch_func(row, df_save)
                    except ValueError as error:
                        print(error)
                        time.sleep(1)
                        continue
                    except OSError:
                        bad_file = True
                        print('Bad file')
                        break
                    break
                if bad_file:
                    continue
 
        print('saving')
        for key in df_save:
            if not df_save[key].empty:
                df_save[key].to_csv(save_path + i.strftime("%Y%m%d") + '_' + key + '.csv', index=False)


def fetch_housekeeping_v1(row, df_save):
    res = requests.get(row['downloadUrl'])
    with io.BytesIO(res.content) as file:
        df = xr.open_dataset(file)
        df_diag = xr.open_dataset(file, group='diagnostics')
        diag = pd.DataFrame([df_diag.attrs])
        diag['datetime'] = df.time[0].values
        df_save['diag'] = pd.concat([df_save['diag'], diag])
        return df_save
        

def fetch_housekeeping_v2(row, df_save):
    res = requests.get(row['downloadUrl'])
    with io.BytesIO(res.content) as file:
        df_monitoring = xr.open_dataset(file, group='monitoring')
        df_status = xr.open_dataset(file, group='status')
        monitoring = df_monitoring.to_dataframe().reset_index()
        monitoring = monitoring.rename({'time': 'datetime'}, axis=1)
        
        status = df_status.to_dataframe().reset_index()
        status = status.rename({'time': 'datetime'}, axis=1)
        
        df_save['monitoring'] = pd.concat([df_save['monitoring'], monitoring])
        df_save['status'] = pd.concat([df_save['status'], status])
        return df_save


def fetch_housekeeping_v3(row, df_save):
    # For Vehmasmaki 2023
    res = requests.get(row['downloadUrl'])
    with io.BytesIO(res.content) as file:
        df_monitoring = xr.open_dataset(file, group='monitoring')
        df_status = xr.open_dataset(file, group='status')
        monitoring = pd.DataFrame([df_monitoring.attrs])
        monitoring = monitoring.rename({'Timestamp': 'datetime'}, axis=1)
        monitoring.datetime = monitoring.datetime.astype(float)
        monitoring['datetime'] = pd.to_datetime(monitoring['datetime'], unit='s')
        
        status = pd.DataFrame([df_status.attrs])
        status = status.rename({'Timestamp': 'datetime'}, axis=1)
        status.datetime = status.datetime.astype(float)
        status['datetime'] = pd.to_datetime(status['datetime'], unit='s')
        
        df_save['monitoring'] = pd.concat([df_save['monitoring'], monitoring])
        df_save['status'] = pd.concat([df_save['status'], status])
        return df_save
