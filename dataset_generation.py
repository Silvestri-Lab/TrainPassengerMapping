### Spark setup

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, FloatType, StructType, ArrayType
import datetime
from functools import reduce
import random 
import datetime

# set seed for reproducibility
random.seed(42)

spark = SparkSession.builder \
    .appName("Railways Traffic Analysis") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.conf.set('spark.sql.session.timeZone', 'CET') 
spark.conf.set('spark.sql.repl.eagerEval.enabled', True) 
spark.conf.set('spark.sql.repl.eagerEval.maxNumRows', 10) 
spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', True) 


### formatting trains for Suffix Tree algorithm building phase

# read the csv file
df = spark.read.csv('suffix_tree_input.csv', header=True) #0
window_train = Window.partitionBy('codice_treno').orderBy('arrivo_teorico')

# preproccesing for the train 
df_cleaned_trains = (
    df
    .withColumn('arrivo_teorico', f.col('arrivo_teorico').cast('int')).withColumn('partenza_teorica', f.col('partenza_teorica').cast('int'))
    .withColumn('arrivo_teorico', f.when(f.col('arrivo_teorico') == 0, f.col('partenza_teorica')).otherwise(f.col('arrivo_teorico')))
    .withColumn('partenza_teorica', f.when(f.col('partenza_teorica') == 0, f.col('arrivo_teorico')).otherwise(f.col('partenza_teorica')))
    .withColumn('od_date_time_start', f.first('partenza_teorica').over(window_train))
    .withColumn('second_value', f.lead('partenza_teorica', 1).over(window_train))
    .withColumn('origin', f.first('station_id').over(window_train))
    .withColumn('destination', f.last('station_id').over(window_train))
    .select('codice_treno', 'od_date_time_start', 'second_value', 'origin', 'destination', 'stop_order')
    .filter(f.col('stop_order') == 1)
    .na.drop() 
    .distinct()
    .groupBy('od_date_time_start', 'second_value','origin', 'destination')
    .agg(f.collect_list('codice_treno').alias('codice_treno'))
    .filter(f.col('od_date_time_start') != 0)
    .withColumn('codice_treno', f.col('codice_treno')[0])
    .select('codice_treno')
    .distinct()
)

df = (
    df
    .withColumn('arrivo_teorico', f.col('arrivo_teorico').cast('int')).withColumn('partenza_teorica', f.col('partenza_teorica').cast('int'))
    .withColumn('arrivo_teorico', f.when(f.col('arrivo_teorico') == 0, f.col('partenza_teorica')).otherwise(f.col('arrivo_teorico')))
    .withColumn('partenza_teorica', f.when(f.col('partenza_teorica') == 0, f.col('arrivo_teorico')).otherwise(f.col('partenza_teorica')))
    .join(df_cleaned_trains, on=['codice_treno'], how='inner') # clean identical trains with different train ids 
    .select('categoria', 'codice_treno', 'station_id', 'arrivo_teorico', 'partenza_teorica', 'stop_order', 'stazione')
    .dropDuplicates()
    .orderBy('codice_treno', 'arrivo_teorico')


)
df.toPandas().to_csv('experiment_input_train_stt_synthetic.csv', index=False)

# Synthetic users without train transfer PERFECT USERS and GROUND TRUTH generation

window_train = Window.partitionBy('codice_treno')

mean = 10
stddev = 10

#1
df_train_with_delay = (
    df
    # find the maximum stop_order for inside the window
    .withColumn('stop_count', f.max('stop_order').over(window_train))
    .select('categoria', 'codice_treno', 'stop_count')
    .distinct()
    # Generate a column 'delay' from a normal distribution with mean 0 and std 1
    .withColumn('delay', f.randn() * stddev + mean)
    .withColumn('delay', f.when(f.col('delay') < 0, 0).when(f.col('delay') > 60, 60).otherwise(f.col('delay')))
)

n = 1
#2
df_user_od = (
    df_train_with_delay
    .select('codice_treno', 'delay','stop_count')
    # duplicate each column n
    .withColumn('dummy', f.explode(f.array([f.lit(x) for x in range(n)])))
    .drop('dummy')
    .withColumn('IMSI', (f.rand(seed=42)*10000).cast('int'))
    # with origin select a number between 0 and stop_count-2
    .withColumn('origin', f.floor(f.rand(seed=42) * (f.col('stop_count') - 1)))
    .withColumn('origin', f.when(f.col('origin') < 1, 1).otherwise(f.col('origin')))
    # with destination select a number between origin+1 and stop_count-1
    .withColumn('destination', f.col('origin') + 1 + f.floor(f.rand(seed=42) * (f.col('stop_count') - f.col('origin') - 1)))
    .select('IMSI', 'codice_treno', 'origin', 'destination', 'delay')
    .withColumnRenamed('origin', 'origin_idx')
    .withColumnRenamed('destination', 'destination_idx')
)

window_user = Window.partitionBy('IMSI','codice_treno')
#3 
ground_truth = (
    df_user_od
    .join(df, on='codice_treno', how='inner')
    # select only the rows between origin_idx and destination_idx in thw window_user
    #.withColumn('stop_order', f.row_number().over(window_user))
    .filter(f.col('stop_order') >= f.col('origin_idx'))
    .filter(f.col('stop_order') <= f.col('destination_idx'))
    .withColumn('arrivo_teorico', f.col('arrivo_teorico').cast('int'))
    .withColumn('partenza_teorica', f.col('partenza_teorica').cast('int'))
    .orderBy('IMSI', 'codice_treno', 'stop_order')
    .withColumn('origin_name', f.first('stazione').over(window_user))
    .withColumn('destination_name', f.last('stazione').over(window_user))
    .withColumn('origin', f.first('station_id').over(window_user))
    .withColumn('destination', f.last('station_id').over(window_user))
    .withColumn('od_date_time_start', f.first('partenza_teorica').over(window_user))
    .withColumn('od_date_time_end', f.last('arrivo_teorico').over(window_user))
    .distinct()
    .filter(f.col('origin') != f.col('destination'))
    .filter(f.col('od_date_time_start') < f.col('od_date_time_end'))
)

ground_truth.toPandas().to_csv('experiment_synthetic_user_ground_truth.csv', index=False) 
# create a lookup table for the train_id and the stat

sytnethic_user_input_stt = (
    ground_truth
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'stop_order', 'date_time_start')
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .dropDuplicates(['IMSI', 'ORIGIN', 'DESTINATION']) 
    .limit(10000)
    .cache()
)

sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic.csv', index=False)


# Synthetic users without train transfer TEMPORAL NOISE

synthetic_user = spark.read.csv('./experiment_synthetic_user_ground_truth.csv', header=True)

window_noiser = Window.partitionBy('IMSI','codice_treno').orderBy(f.rand())

sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'stop_order', 'date_time_start')
    # cast dealy column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    # add dealay to the date_time_start, dealys is in minutes and date_time_start is in seconds
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START') + f.col('delay') * 60)
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END') + f.col('delay') * 60)
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START').cast('int'))
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END').cast('int'))
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .dropDuplicates(['IMSI', 'ORIGIN', 'DESTINATION']) 
    .limit(10000)
    .repartition(1)
    .cache()
)

sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic_noise_only_time.csv', index=False)


# Synthetic users without train transfer SPATIAL NOISE (deleting stations)

synthetic_user = spark.read.csv('./experiment_synthetic_user_ground_truth.csv', header=True)

sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'stop_order', 'date_time_start')
    # cast delay column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    # add delay to the date_time_start, dealys is in minutes and date_time_start is in seconds
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('STATIONS', f.when(f.size(f.col('STATIONS')) > 2, f.expr('transform(STATIONS, (x, i) -> if(i != 0 and i != size(STATIONS) - 1 and rand() > 0.5, null, x))')).otherwise(f.col('STATIONS')))
    # find the index of the null values in the list of stations and drop the corresponding values in the list of dates
    .withColumn('DATES_TIME_START', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_START[i]))'))
    .withColumn('DATES_TIME_END', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_END[i]))'))
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .dropDuplicates(['IMSI', 'ORIGIN', 'DESTINATION']) 
    .limit(10000)
    .repartition(1)
    .cache()
)
sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic_noise_only_spatial_(stations_removal).csv', index=False)

# Synthetic users without train transfer SPATIAL NOISE (adding noise stations)

# Define the function to insert noises
def insert_noise_station(stations, idx, noise_flag):
    if noise_flag and idx != 0:
        random_station = -7
        stations.insert(idx, random_station)
    return stations

def insert_noise_time(time, idx, noise_flag):
    if noise_flag and idx != 0:
        time_to_insert = int((time[idx - 1] + time[idx]) / 2)
        time.insert(idx, time_to_insert)
    return time

# Register the UDF
insert_noise_station_udf = f.udf(insert_noise_station, ArrayType(IntegerType()))
insert_noise_time_udf = f.udf(insert_noise_time, ArrayType(IntegerType()))


# Read the CSV file
synthetic_user = spark.read.csv('./synthetic_user_ground_truth_bruno_method.csv', header=True)

# Process the DataFrame
sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'stop_order', 'DATE_TIME_START')
    .withColumn('delay', f.col('delay').cast('int'))
    .withColumn('DATE_TIME_START', (f.col('DATE_TIME_START') + f.col('delay') * 60).cast('int'))
    .withColumn('DATE_TIME_END', (f.col('DATE_TIME_END') + f.col('delay') * 60).cast('int'))
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), 
         f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), 
         f.collect_list('DATE_TIME_END').alias('DATES_TIME_END')) 
    .withColumn('NOISE', f.rand() <= 0.5)
    .withColumn('idx', f.floor(f.rand() * f.size(f.col('STATIONS'))).cast('int'))
    # cast stations to array type of integer
    .withColumn('STATIONS', f.col('STATIONS').cast(ArrayType(IntegerType())))
    .withColumn('STATIONS', insert_noise_station_udf(f.col('STATIONS'), f.col('idx'), f.col('NOISE')))
    .withColumn('DATES_TIME_START', f.col('DATES_TIME_START').cast(ArrayType(IntegerType())))
    .withColumn('DATES_TIME_START', insert_noise_time_udf(f.col('DATES_TIME_START'), f.col('idx'), f.col('NOISE')))
    .withColumn('DATES_TIME_END', f.col('DATES_TIME_END').cast(ArrayType(IntegerType())))
    .withColumn('DATES_TIME_END', insert_noise_time_udf(f.col('DATES_TIME_END'), f.col('idx'), f.col('NOISE'))) 
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .drop('NOISE', 'idx')
    .repartition(1)
    .cache()
)

# save the result
sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic_noise_only_spatial_(stations_noise_add).csv', index=False)


# Synthetic users without train transfer COMBINED NOISE

synthetic_user = spark.read.csv('./experiment_synthetic_user_ground_truth.csv', header=True)

window_noiser = Window.partitionBy('IMSI','codice_treno').orderBy(f.rand())

sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'stop_order', 'date_time_start')
    # cast dealy column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START') + f.col('delay') * 60)
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END') + f.col('delay') * 60)
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START').cast('int'))
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END').cast('int'))
    # add dealay to the date_time_start, dealys is in minutes and date_time_start is in seconds
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('STATIONS', f.when(f.size(f.col('STATIONS')) > 2, f.expr('transform(STATIONS, (x, i) -> if(i != 0 and i != size(STATIONS) - 1 and rand() > 0.5, null, x))')).otherwise(f.col('STATIONS')))
    # find the index of the null values in the list of stations and drop the corresponding values in the list of dates
    .withColumn('DATES_TIME_START', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_START[i]))'))
    .withColumn('DATES_TIME_END', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_END[i]))'))
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .dropDuplicates(['IMSI', 'ORIGIN', 'DESTINATION']) 
    .limit(10000)
    .repartition(1)
    .cache()
)

sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic_noise_spatial_temporal.csv', index=False)


# Synthetic users with train transfer PERFECT USER
mean = 5
stddev = 5
# read the csv file
synthetic_perfect_user_with_change_train = spark.read.csv('experiment_input_train_stt_synthetic.csv', header=True) #0
window_train = Window.partitionBy('codice_treno').orderBy('arrivo_teorico')


station_registry = spark.read.csv('experiment_station_registry_change_station.csv', header=True).filter(f.col('CHANGE_STATION') == True).select('STATION_ID')

departure_trains = (
    synthetic_perfect_user_with_change_train
    .join(station_registry, on='STATION_ID', how='inner')
    .filter(f.col('stop_order') == 1)
    .withColumnRenamed('codice_treno', 'departure_train_id')
    .withColumnRenamed('stop_order', 'departure_stop_order')
    .withColumnRenamed('arrivo_teorico', 'departure_train_arrival_time')
    .withColumnRenamed('partenza_teorica', 'departure_train_departure_time')
    .select('STATION_ID', 'departure_train_id', 'departure_stop_order', 'departure_train_arrival_time', 'departure_train_departure_time')
)

arrival_trains = (
    synthetic_perfect_user_with_change_train
    .join(station_registry, on='STATION_ID', how='inner')
    .filter(f.col('stop_order') != 1)
    .withColumnRenamed('codice_treno', 'arrival_train_id')
    .withColumnRenamed('stop_order', 'arrival_stop_order')
    .withColumnRenamed('arrivo_teorico', 'arrival_train_arrival_time')
    .withColumnRenamed('partenza_teorica', 'arrival_train_departure_time')
    .select('STATION_ID', 'arrival_train_id', 'arrival_stop_order', 'arrival_train_arrival_time', 'arrival_train_departure_time','stazione')
)

arrival_departure_trains = (
    departure_trains
    .join(arrival_trains, on='STATION_ID', how='inner')
    .filter(f.col('departure_stop_order') < f.col('arrival_stop_order'))
    .filter(f.col('departure_train_id') != f.col('arrival_train_id'))
    .withColumn('time_diff', (f.col('departure_train_departure_time') - f.col('arrival_train_arrival_time')) / 60)
    .filter((f.col('time_diff') >= 5) & (f.col('time_diff') <= 60))
    .select('station_id', 'departure_train_id', 'departure_stop_order', 'arrival_train_id', 'arrival_stop_order', 'time_diff')
    .distinct()
    # gemerate an id for each row
    .withColumn('id', f.monotonically_increasing_id())
    .withColumnRenamed('station_id', 'station_id_arrival_departure_synthetic_perfect_user_with_change_train')
    .cache()
)

change_train_journey = (
    arrival_departure_trains
    # join on synthetic_perfect_user_with_change_train with codice_treno and departure_train_id
    .join(synthetic_perfect_user_with_change_train, (f.col('codice_treno') == f.col('departure_train_id')), how='inner')
    .orderBy('id', 'arrivo_teorico')
    .select('id','codice_treno','arrivo_teorico', 'partenza_teorica', 'stop_order','station_id', 'stazione')
    .withColumn('change_train', f.lit(True))
)

pre_change_train_journey = (
    arrival_departure_trains
    # join on synthetic_perfect_user_with_change_train with codice_treno and arrival_train_id
    .join(synthetic_perfect_user_with_change_train, (f.col('codice_treno') == f.col('arrival_train_id')), how='inner')
    .orderBy('id', 'arrivo_teorico')
    .select('id','codice_treno','arrivo_teorico', 'partenza_teorica', 'stop_order','station_id', 'stazione')
    .withColumn('change_train', f.lit(False))
    #.filter(f.col('stop_order') <= f.col('arrival_stop_order'))
)

# make a window over id 
window = Window.partitionBy('id').orderBy('arrivo_teorico')
windo_without_order = Window.partitionBy('id')

    
synthetic_perfect_user_with_change_train = (
    # concat change_train_journey and pre_change_train_journey
    change_train_journey
    .union(pre_change_train_journey)
    .orderBy('id', 'arrivo_teorico')
    # when the previous line is different from the current line for the chaang_train values modify the current line with the previous arrivo_teorico
    .withColumn('partenza_teorica', f.when(f.lead('change_train').over(window) != f.col('change_train'), f.lead('partenza_teorica').over(window)).otherwise(f.col('partenza_teorica')))
    # change station
    .withColumn('change_station', f.when(f.lead('change_train').over(window) != f.col('change_train'), f.lead('station_id').over(window)).otherwise(None))
    .withColumn('change_station', f.last('change_station', True).over(windo_without_order))
    .filter(~((f.col('stop_order') == 1) & (f.col('change_train') == True)))
    # take the last value of station_id for change_train value true over the window
    .withColumn('stop_order_unified', f.row_number().over(window))
)

# Define windows specification
window_spec = Window.partitionBy("id").orderBy("stop_order")
window_partition = Window.partitionBy("id").orderBy("arrivo_teorico")
window_train = Window.partitionBy("id","codice_treno").orderBy("arrivo_teorico")


# Mark rows where station_id equals change_station
synthetic_perfect_user_with_change_train = synthetic_perfect_user_with_change_train.withColumn("is_change_station", f.col("station_id") == f.col("change_station"))

# Create columns for rows before and after the change station
synthetic_perfect_user_with_change_train = (
    synthetic_perfect_user_with_change_train
    .withColumn("row_idx", f.row_number().over(window_spec))
    .withColumn("lag_1", f.lag("is_change_station", 1).over(window_spec))
    .withColumn("lag_2", f.lag("is_change_station", 2).over(window_spec))
    .withColumn("lag_3", f.lag("is_change_station", 3).over(window_spec))
    .withColumn("lag_4", f.lag("is_change_station", 4).over(window_spec))
    .withColumn("lead_1", f.lead("is_change_station", 1).over(window_spec))
    .withColumn("lead_2", f.lead("is_change_station", 2).over(window_spec))
    .withColumn("lead_3", f.lead("is_change_station", 3).over(window_spec))
    .withColumn("lead_4", f.lead("is_change_station", 4).over(window_spec))
)

# Filter the rows within the range of 4 rows before and after the change station
filtered_df = synthetic_perfect_user_with_change_train.filter(
    f.col("is_change_station") |
    f.col("lag_1") | f.col("lag_2") | f.col("lag_3") | f.col("lag_4") |
    f.col("lead_1") | f.col("lead_2") | f.col("lead_3") | f.col("lead_4")
).orderBy("id", "arrivo_teorico")

synthetic_perfect_user_with_change_train = (
    filtered_df
    # randomly delete a row with lag_1 == True
    .withColumn("delete", f.when(f.col("lag_1"), f.rand() < 0.5).otherwise(False))
    # randomly delete a row with lead_4 == True
    .withColumn("delete", f.when(f.col("lead_4"), f.rand() < 0.5).otherwise(f.col("delete")))
    .filter(~f.col("delete")) 
    .withColumn('origin', f.first('station_id').over(window_partition))
    .withColumn('last_value', f.row_number().over(window_partition))
    # use max last_value over the window partition to get the value of station_id
    .withColumn('destination', f.when(f.col('last_value') == f.max('last_value').over(windo_without_order), f.col('station_id')).otherwise(None))
    # fill the none value in the windo_without_order with the valid value of destination
    .withColumn('destination', f.last('destination', True).over(windo_without_order))
    .withColumn('OD_DATE_TIME_START', f.when(f.col('origin') == f.col('station_id'), f.col('arrivo_teorico')).otherwise(None))
    .withColumn('OD_DATE_TIME_START', f.first('OD_DATE_TIME_START', True).over(windo_without_order))
    .withColumn('OD_DATE_TIME_END', f.when(f.col('destination') == f.col('station_id'), f.col('partenza_teorica')).otherwise(None))
    .withColumn('OD_DATE_TIME_END', f.last('OD_DATE_TIME_END', True).over(windo_without_order))
    .distinct()
    .filter(~((f.col('arrivo_teorico') == 0) & (f.col('partenza_teorica') == 0)))
    .select('id', 'codice_treno', 'arrivo_teorico', 'partenza_teorica', 'station_id', 'stazione', 'change_train', 'change_station', 'stop_order_unified', 'origin', 'destination', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .filter((f.col('OD_DATE_TIME_START') != 0) & (f.col('OD_DATE_TIME_END') != 0))
    .filter(f.col('origin') != f.col('destination'))
    .withColumn('stop_order', f.row_number().over(window_partition))
    # apply for each train_partition using a window a random delay
    .withColumn('Rank', f.row_number().over(window_train))
    .withColumn('delay', f.when(f.col('Rank') == 1, f.randn() * stddev + mean).otherwise(None))
    .withColumn('delay', f.last('delay', True).over(window_train))
    .withColumn('delay', f.when(f.col('delay') < 0, 0).when(f.col('delay') > 10, 10).otherwise(f.col('delay')))
    .orderBy('id','arrivo_teorico','stop_order_unified')
    .cache()

)

filtered_df
synthetic_perfect_user_with_change_train.toPandas().to_csv('experiment_synthetic_user_ground_truth_with_change_train.csv', index=False)

### Synthetic users with train transfer TEMPORAL NOISE
synthetic_user = spark.read.csv('./experiment_synthetic_user_ground_truth_with_change_train.csv', header=True)

window_noiser = Window.partitionBy('id','codice_treno').orderBy(f.rand())

sytnethic_user_input_stt_with_change = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('id', 'codice_treno', 'stop_order', 'DATE_TIME_START')  
    # cast dealy column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    # add dealay to the DATE_TIME_START, dealys is in minutes and DATE_TIME_START is in seconds
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START') + f.col('delay') * 60)
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END') + f.col('delay') * 60)
    .withColumn('DATE_TIME_START', f.col('DATE_TIME_START').cast('int'))
    .withColumn('DATE_TIME_END', f.col('DATE_TIME_END').cast('int'))
    .groupBy('DATE_ID', 'id', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .filter((f.col('STATIONS_LEN') <= 8) & (f.col('STATIONS_LEN') > 3))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .dropDuplicates(['id', 'ORIGIN', 'DESTINATION']) 
    .limit(10000)
    .repartition(1)
    .cache()
)
sytnethic_user_input_stt_with_change.toPandas().to_csv('experiment_input_user_stt_synthetic_with_change_noise_temporal.csv', index=False)

### Synthetic users with train transfer SPATIAL NOISE (stations removal)
synthetic_user = spark.read.csv('./synthetic_user_ground_truth_bruno_method_with_change_train_with_delay.csv', header=True)

window_noiser = Window.partitionBy('IMSI','codice_treno').orderBy(f.rand())

df_result_vanilla = spark.read.csv('./synthetic_result_user_vanilla_with_change.csv', header=True).select('IMSI').distinct()

sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('id', 'IMSI')
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'date_time_start')
    # cast dealy column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    # add dealay to the date_time_start, dealys is in minutes and date_time_start is in seconds
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('STATIONS', f.when(f.size(f.col('STATIONS')) > 2, f.expr('transform(STATIONS, (x, i) -> if(i != 0 and i != size(STATIONS) - 1 and rand() > 0.8, null, x))')).otherwise(f.col('STATIONS')))
    # find the index of the null values in the list of stations and drop the corresponding values in the list of dates
    .limit(10000)
    .withColumn('DATES_TIME_START', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_START[i]))'))
    .withColumn('DATES_TIME_END', f.expr('transform(sequence(0, size(STATIONS) - 1), i -> if(STATIONS[i] is null, null, DATES_TIME_END[i]))'))
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .join(df_result_vanilla, on='IMSI', how='inner')
    .select('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END', 'STATIONS_ALL', 'DATES_TIME_START_ALL', 'DATES_TIME_END_ALL')
)

sytnethic_user_input_stt.toPandas().to_csv('experiment_input_user_stt_synthetic_with_change_noise_spatial_(stations_removal).csv', index=False)

# Define the function to insert noises
def insert_noise_station(stations, idx, noise_flag):
    if noise_flag and idx != 0:
        random_station = -7
        stations.insert(idx, random_station)
    return stations

def insert_noise_time(time, idx, noise_flag):
    if noise_flag and idx != 0:
        time_to_insert = int((time[idx - 1] + time[idx]) / 2)
        time.insert(idx, time_to_insert)
    return time

# Register the UDF
insert_noise_station_udf = f.udf(insert_noise_station, ArrayType(IntegerType()))
insert_noise_time_udf = f.udf(insert_noise_time, ArrayType(IntegerType()))

synthetic_user = spark.read.csv('./synthetic_user_ground_truth_bruno_method_with_change_train_with_delay.csv', header=True)

window_noiser = Window.partitionBy('IMSI','codice_treno').orderBy(f.rand())

df_result_vanilla = spark.read.csv('./synthetic_result_user_vanilla_with_change.csv', header=True).select('IMSI').distinct()

sytnethic_user_input_stt = (
    synthetic_user
    .withColumn('DATE_ID', f.date_format(f.to_date(f.lit(datetime.datetime.now())), 'yyyyMMdd'))
    .withColumnRenamed('id', 'IMSI')
    .withColumnRenamed('station_id', 'STATION')
    .withColumnRenamed('arrivo_teorico', 'DATE_TIME_START')
    .withColumnRenamed('partenza_teorica', 'DATE_TIME_END')
    .orderBy('IMSI', 'codice_treno', 'date_time_start')
    # cast dealy column to int 
    .withColumn('delay', f.col('delay').cast('int'))
    # add dealay to the date_time_start, dealys is in minutes and date_time_start is in seconds
    .groupBy('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END')
    .agg(f.collect_list('STATION').alias('STATIONS'), f.collect_list('DATE_TIME_START').alias('DATES_TIME_START'), f.collect_list('DATE_TIME_END').alias('DATES_TIME_END'))
    # if the origini is different from the first position of the list of stations and the destination has to be equal to the last position of the list of stations
    .filter((f.col('ORIGIN') == f.col('STATIONS')[0]) & (f.col('DESTINATION') == f.col('STATIONS')[f.size(f.col('STATIONS')) - 1]))
    # drop a random STATIONS in the list, only for stations with more than 2 stations, it doens't have to be the first or the last in the station list
    .withColumn('NOISE', f.rand() <= 0.5)
    .withColumn('idx', f.floor(f.rand() * f.size(f.col('STATIONS'))).cast('int'))
    .withColumn('STATIONS', f.col('STATIONS').cast(ArrayType(IntegerType())))
    .withColumn('STATIONS', insert_noise_station_udf(f.col('STATIONS'), f.col('idx'), f.col('NOISE')))
    .withColumn('DATES_TIME_START', f.col('DATES_TIME_START').cast(ArrayType(IntegerType())))
    .withColumn('DATES_TIME_START', insert_noise_time_udf(f.col('DATES_TIME_START'), f.col('idx'), f.col('NOISE')))
    .withColumn('DATES_TIME_END', f.col('DATES_TIME_END').cast(ArrayType(IntegerType())))
    .withColumn('DATES_TIME_END', insert_noise_time_udf(f.col('DATES_TIME_END'), f.col('idx'), f.col('NOISE'))) 
    .withColumn('STATIONS', f.concat_ws('|', 'STATIONS'))
    .withColumn('DATES_TIME_START', f.concat_ws('|', 'DATES_TIME_START'))
    .withColumn('DATES_TIME_END', f.concat_ws('|', 'DATES_TIME_END')) 
    .withColumnRenamed('STATIONS', 'STATIONS_ALL')
    .withColumnRenamed('DATES_TIME_START', 'DATES_TIME_START_ALL')
    .withColumnRenamed('DATES_TIME_END', 'DATES_TIME_END_ALL')
    .join(df_result_vanilla, on='IMSI', how='inner')
    .select('DATE_ID', 'IMSI', 'ORIGIN', 'DESTINATION', 'OD_DATE_TIME_START', 'OD_DATE_TIME_END', 'STATIONS_ALL', 'DATES_TIME_START_ALL', 'DATES_TIME_END_ALL')
    .limit(10000)
)

sytnethic_user_input_stt.toPandas().to_csv('input_user_stt_synthetic_witch_change_noise_only_spatial_(add_noise_stations).csv', index=False)

