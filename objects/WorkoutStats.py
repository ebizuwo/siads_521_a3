import numpy as np
import pandas as pd


def diff(s):
    return s.iloc[-1] - s.iloc[0]


def drop_nan(x):
    return x[~np.isnan(x)]

def modify_df(strava):
    # convert time stamp
    strava['timestamp'] = pd.to_datetime(strava['timestamp'])
    # do a little work to map a custom
    wo_id = strava['datafile'].unique()
    strava['workout'] = strava['datafile']
    strava.workout = strava.workout.map({n: i for i, n in enumerate(wo_id)})
    # modify lat_long
    strava.position_lat *= (180 / 2 ** 31)
    strava.position_long *= (180 / 2 ** 31)

    # create delta altitude
    strava['delta_altitude'] = strava.altitude.shift() - strava.altitude
    return strava


class WorkoutStats:

    def __init__(self, df):
        self.df = modify_df(df)
        self.wo_num = np.array([wo for wo in self.df.workout.unique()])
        self.wo_dfs = [self.df[self.df.workout == wo] for wo in self.wo_num]
        # workout start times used as the monthly metric tracking
        self.x = np.array([wo.timestamp.iloc[0] for wo in self.wo_dfs])
        self.distance = self.Distance(self.wo_dfs, self.x)
        self.time = self.Time(self.wo_dfs, self.x)
        self.speed = self.Speed(self.wo_dfs, self.x)
        self.power = self.Power(self.wo_dfs, self.x)
        self.heartrate = self.HeartRate(self.wo_dfs, self.x)
        self.cadence = self.Cadence(self.wo_dfs, self.x)
        self.altitude = self.Altitude(self.wo_dfs, self.x)
        self.latlong = self.LatLong(self.wo_dfs, self.x)
        self.metrics = ['Distance', 'Time', 'Speed', 'Power', 'Heartrate', 'Cadence', 'Altitude']
        self.geographic = ['Latitude', 'Longitude']

    def trend_df(self):
        from functools import reduce
        data = [
            self.distance.xy_avg(),
            self.time.xy_time_duration(),
            self.speed.xy_avg(),
            self.power.xy_avg(),
            self.heartrate.xy_avg(),
            self.cadence.xy_avg(),
            self.altitude.xy_avg(),
            self.altitude.xy_avg_delta(),
            pd.DataFrame({'timestamp': self.x, 'workout': self.wo_num+1})
        ]
        return reduce(lambda l, r: pd.merge(l, r, on='timestamp', how='inner'), data)

    def full_df(self):
        return self.df


    def avg_gen_stats(self):
        return {
            'metric':{
            'total_distance': self.distance.total(),
            'max_dist': self.distance.max_dist(),
            'min_dist': self.distance.min_dist(),
            'avg_workout_time': self.time.avg(),
            'avg_speed' : self.speed.avg(),
            'avg_power' : self.power.avg(),
            'avg_heartrate': self.heartrate.avg(),
            'avg_cadence': self.cadence.avg(),
            'avg' : self.altitude.avg(),
            'uphill': self.altitude.total_altitude_gain_loss()[0],
            'downhill': self.altitude.total_altitude_gain_loss()[1],
            }
        }

    def workout_by_num(self, wo_num):
        return{
            'metric':{
                'traveled': self.distance.dist_wo(wo_num),
                'workout_time': self.time.workout_time(wo_num),
                'avg_speed': self.speed.workout_avg_speed(wo_num),
                'top_speed': self.speed.workout_top_speed(wo_num),
                'low_speed': self.speed.workout_min_speed(wo_num),
                'avg_power': self.power.workout_avg_power(wo_num),
                'avg_heartrate': self.heartrate.workout_avg_heartrate(wo_num),
                'peak_heartrate': self.heartrate.workout_peak_heartrate(wo_num),
                'avg_cadence' : self.cadence.workout_avg_cadence(wo_num),
                'peak_cadence': self.cadence.workout_top_cadence(wo_num)
            }
        }


    class WholePic:

        def __init__(self, df):
            self.df = df

    class Distance:
            def total(self):
                return self.dist.sum().round(2)

            def dist_wo(self, wo_num):
                return self.wo_dfs[wo_num].distance.iloc[-1]

            def max_dist(self):
                return max(self.dist)

            def min_dist(self):
                return min(self.dist)

            def workout_distance(self, wo_num):
                wo = self.wo_dfs[wo_num]
                return wo.timestamp.iloc[0], wo.distance.iloc[-1]

            def xy_avg(self):
                # uses start time of the workout
                ts = []
                ds = []
                for wo in self.wo_dfs:
                    df = wo[self.cols]
                    d = df.distance.iloc[-1]
                    t = df.timestamp.iloc[0]
                    ds.append(d)
                    ts.append(t)
                return pd.DataFrame({'timestamp': ts, 'distance': ds})

            def __init__(self, wo_dfs, x):
                self.wo_dfs = wo_dfs
                self.cols = ['timestamp', 'distance']
                self.dist = np.array([wo.distance.iloc[-1] for wo in self.wo_dfs])
                self.x = x

    class Time:
        def avg(self):
            return self.tdur.mean()

        def workout_time(self, wo_num):
            return self.tdur[wo_num]

        def xy_time_duration(self):
            ts = []
            tds = []
            for wo in self.wo_dfs:
                t = wo.timestamp.iloc[0]
                td = diff(wo.timestamp).total_seconds()
                ts.append(t)
                tds.append(td)
            return pd.DataFrame(data={'timestamp': ts, 'time_duration': tds})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.tdur = np.array([diff(wo.timestamp) for wo in self.wo_dfs])
            self.x = x

    class Speed:
        def avg(self):
            return drop_nan(self.speed).mean().round(2)

        def workout_avg_speed(self, wo_num):
            # wo = self.wo_dfs[wo_num]
            avgspd = self.speed[wo_num].round(2)
            return avgspd

        def workout_top_speed(self, wo_num):
            ts = [wo.speed.max() for wo in self.wo_dfs]
            return ts[wo_num]

        def workout_min_speed(self, wo_num):
            ts = [wo.speed.min() for wo in self.wo_dfs]
            return ts[wo_num]

        def xy_avg(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_speed':self.speed})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.speed = np.array([wo.speed.mean() for wo in self.wo_dfs])
            self.x = x

    class Power:
        def avg(self):
            return drop_nan(self.power).mean().round(2)

        def workout_avg_power(self, wo_num):
            wo = self.wo_dfs[wo_num]
            avgpow = self.power[wo_num].round(2)
            return avgpow

        def xy_avg(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_power':self.power})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.power = np.array([wo.Power.mean() for wo in self.wo_dfs])
            self.x = x

    class HeartRate:
        def avg(self):
            return drop_nan(self.hr).mean().round(2)

        def workout_avg_heartrate(self, wo_num):
            wo = self.wo_dfs[wo_num]
            avg_hr = self.hr[wo_num].round(2)
            return avg_hr

        def workout_peak_heartrate(self, wo_num):
            wo = [wo.heart_rate.max() for wo in self.wo_dfs]
            return wo[wo_num]


        def xy_avg(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_heart_rate': self.hr})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.hr = np.array([wo.heart_rate.mean() for wo in self.wo_dfs])
            self.x = x

    class Cadence:
        def avg(self):
            return drop_nan(self.cadence).mean().round(2)

        def workout_avg_cadence(self, wo_num):
            avg_cad = [wo.cadence.mean() for wo in self.wo_dfs]
            return avg_cad[wo_num]

        def workout_top_cadence(self, wo_num):
            top_cad = [wo.cadence.max() for wo in self.wo_dfs]
            return top_cad[wo_num]

        def xy_avg(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_cadence': self.cadence})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.cadence = np.array([wo.cadence.mean() for wo in self.wo_dfs])
            self.x = x


    class Altitude:
        def avg(self):
            return drop_nan(self.altitude).mean().round(2)

        def total_altitude_gain_loss(self):
            dh = []
            uph = []
            for wo in self.wo_dfs:
                for d in wo.delta_altitude:
                    if d < 0:
                        dh.append(d)
                    if d >= 0:
                        uph.append(d)
            return sum(uph), sum(dh)
        # TODO: Fix Max Elevation
        def max_elevation(self, wo_num):
            pass

        def xy_avg(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_altitude': self.altitude})

        def xy_avg_delta(self):
            return pd.DataFrame(data={'timestamp': self.x, 'avg_altitude_delta': self.altitude_delta})

        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.altitude = np.array([wo.altitude.mean() for wo in self.wo_dfs])
            self.altitude_delta = np.array([wo.delta_altitude.mean() for wo in self.wo_dfs])
            self.x = x


    class LatLong:
        def __init__(self, wo_dfs, x):
            self.wo_dfs = wo_dfs
            self.x = x

