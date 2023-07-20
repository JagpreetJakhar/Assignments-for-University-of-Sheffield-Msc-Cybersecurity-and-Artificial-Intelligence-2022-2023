from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
import pandas as pd
from pyspark.sql.functions import substring


spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment") \
    .config("spark.local.dir","/fastdata/acq22jj") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.




log_raw = spark.read.text('Data/NASA_access_log_Jul95.gz').cache()

log_file_fmt = log_raw.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()



hosts_de = log_file_fmt.filter(log_file_fmt['host'].endswith('.de')).cache()
hosts_ca = log_file_fmt.filter(log_file_fmt['host'].endswith('.ca')).cache()
hosts_sg = log_file_fmt.filter(log_file_fmt['host'].endswith('.sg')).cache()

print(f'Total Number of requests from Germany:{hosts_de.count()}')
print(f'Total Number of requests from Canada : {hosts_ca.count()}')
print(f'Total Number of requests from Singapore : {hosts_sg.count()}')

#-------------------------------------------------Q1-A-Visualisation-----------------------------#

counts = [hosts_de.count(),hosts_ca.count(),hosts_sg.count()]
range = np.arange(len(counts))
colors = ['black', 'red', 'yellow']
plt.bar(range,counts,color=colors)
plt.xticks(range, ['Germany', 'Canada', 'Singapore'])
for i, v in enumerate(counts):
    plt.text(i, v + 1000, str(v), color='black', ha='center', fontweight='bold')
plt.title('Total Requests')
plt.ylabel('Total Requests')
plt.savefig('Output/Q1_A.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.clf()
#--------------------------------------------------------------------------------------------

#-------------------------------------Q1-B--------------------------------------------------
unique_de = hosts_de.select('host').distinct().count()
unique_ca = hosts_ca.select('host').distinct().count()
unique_sg = hosts_sg.select('host').distinct().count()

print(f'Number of unique hosts from Germany: {unique_de}')
print(f'Number of unique hosts from Candada : {unique_ca}')
print(f'Number of unique hosts from Singapore: {unique_sg}')


frequent_de = hosts_de.select('host').groupBy('host').count().sort('count', ascending=False)
frequent_de = frequent_de.withColumnRenamed("host", "Germany")


frequent_ca = hosts_ca.select('host').groupBy('host').count().sort('count', ascending=False)
frequent_ca = frequent_ca.withColumnRenamed("host", "Canada")


frequent_sg = hosts_sg.select('host').groupBy('host').count().sort('count', ascending=False)
frequent_sg = frequent_sg.withColumnRenamed("host", "Singapore")

top9_de = frequent_de.limit(9)
top9_ca = frequent_ca.limit(9)
top9_sg = frequent_sg.limit(9)


top9_de.show(truncate=False),top9_ca.show(truncate=False),top9_sg.show(truncate=False)


#------------------------------------------------------------------------------------------

#-------------------------------Q1-C------------------------------------------------------

rest_de = frequent_de.exceptAll(top9_de)
rest_ca = frequent_ca.exceptAll(top9_ca)
rest_sg = frequent_sg.exceptAll(top9_sg)

rest_total_de = rest_de.agg({"count": "sum"}).collect()[0][0]
rest_total_ca = rest_ca.agg({"count": "sum"}).collect()[0][0]
rest_total_sg = rest_sg.agg({"count": "sum"}).collect()[0][0]

top9_de_pd = top9_de.toPandas()
top9_de_pd.loc[9] = ['Rest', rest_total_de]
top9_ca_pd = top9_ca.toPandas()
top9_sg_pd = top9_sg.toPandas()
top9_ca_pd.loc[9] = ['Rest', rest_total_ca]
top9_sg_pd.loc[9] = ['Rest', rest_total_sg]



#----------Germany Visualisation: --
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# create the chart
plt.pie(top9_de_pd['count'], labels=top9_de_pd['Germany'], autopct='%1.2f%%', explode=[0.1]*len(top9_de_pd),
        startangle=90, colors=colors, textprops={'fontsize': 8},pctdistance=1.1, labeldistance=1.2)

# add a title
plt.title('Germany Hosts Distribution', fontsize=16,loc='right')

# remove legend
plt.legend().remove()

# set equal axis
plt.axis('equal')

# set size of the chart
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.savefig('Output/Q1_C_Ger.png', dpi=400,bbox_inches='tight')
plt.clf()
# show the chart
#plt.show()


#----------Canada Visualisation---

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# create the chart
plt.pie(top9_ca_pd['count'], labels=top9_ca_pd['Canada'], autopct='%1.2f%%', explode=[0.08]*len(top9_ca_pd),
        startangle=140, colors=colors,textprops={'fontsize': 8},pctdistance=1.1, labeldistance=1.3)

# add a title
plt.title('Canada Hosts Distribution', fontsize=8,loc='right')

# remove legend
plt.legend().remove()

# set equal axis
plt.axis('scaled')

# set size of the chart
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.savefig('Output/Q1_C_Can.png', dpi=400,bbox_inches='tight')
plt.clf()
# show the chart
#plt.show()




#---------------Singapore Visualisation --


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# create the chart
plt.pie(top9_sg_pd['count'], labels=top9_sg_pd['Singapore'], autopct='%1.2f%%', explode=[0.08]*len(top9_sg_pd),
        startangle=45, colors=colors,textprops={'fontsize': 8},pctdistance=1.1, labeldistance=1.3)

# add a title
plt.title('Singapore Hosts Distribution', fontsize=8,loc='right')

# remove legend
plt.legend().remove()

# set equal axis
plt.axis('scaled')

# set size of the chart
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.savefig('Output/Q1_C_Sing.png', dpi=400,bbox_inches='tight')
plt.clf()
# show the chart
#plt.show()




#--------------------------------------------------------------------------------------------------------



#---------Part-D ---------------------------------------------------------------------------------------


# group by host and count the number of occurrences
host_counts_de = hosts_de.select('host').groupBy('host').count().sort(desc('count'))

# extract the most frequent host
most_frequent_host_de = host_counts_de.first()['host']

# filter the original DataFrame based on the most frequent host
most_frequent_host_visits_de = hosts_de.filter(hosts_de['host'] == most_frequent_host_de)

# extract the times and hours of the visits
times_and_hours_de =most_frequent_host_visits_de.select(substring('timestamp', 1, 2).alias('day'), substring('timestamp', 13, 2).alias('hour'))

date_hours_pd_de = times_and_hours_de.toPandas()



# count the number of visits on each date-hour combination
counts_de = date_hours_pd_de.groupby(['day', 'hour']).size().reset_index(name='count')

# create pivot table for heatmap
heatmap_data_de = pd.pivot_table(counts_de, values='count', index=['hour'], columns=['day'])

# plot heatmap
plt.pcolormesh(heatmap_data_de, cmap='Reds')
plt.colorbar()
plt.title(f'Germany Top Host: {most_frequent_host_de}')
plt.xlabel('Date')
plt.ylabel('Hour')
plt.xticks(np.arange(0.5, len(heatmap_data_de.columns), 1), heatmap_data_de.columns, rotation=90)
plt.yticks(np.arange(0.5, len(heatmap_data_de.index), 1), heatmap_data_de.index)
plt.savefig('Output/Q1_D_Ger.png', dpi=400,bbox_inches='tight')
#plt.show()
plt.clf()


#-------Canada : ------------

# group by host and count the number of occurrences
host_counts_ca = hosts_ca.select('host').groupBy('host').count().sort(desc('count'))

# extract the most frequent host
most_frequent_host_ca = host_counts_ca.first()['host']

# filter the original DataFrame based on the most frequent host
most_frequent_host_visits_ca = hosts_ca.filter(hosts_ca['host'] == most_frequent_host_ca)

# extract the times and hours of the visits
times_and_hours_ca =most_frequent_host_visits_ca.select(substring('timestamp', 1, 2).alias('day'), substring('timestamp', 13, 2).alias('hour'))

date_hours_pd_ca = times_and_hours_ca.toPandas()



# count the number of visits on each date-hour combination
counts_ca = date_hours_pd_ca.groupby(['day', 'hour']).size().reset_index(name='count')

# create pivot table for heatmap
heatmap_data_ca = pd.pivot_table(counts_ca, values='count', index=['hour'], columns=['day'])

# plot heatmap
plt.pcolormesh(heatmap_data_ca, cmap='Blues')
plt.colorbar()
plt.title(f'Canada Top Host: {most_frequent_host_ca}')
plt.xlabel('Date')
plt.ylabel('Hour')
plt.xticks(np.arange(0.5, len(heatmap_data_ca.columns), 1), heatmap_data_ca.columns, rotation=90)
plt.yticks(np.arange(0.5, len(heatmap_data_ca.index), 1), heatmap_data_ca.index)
plt.savefig('Output/Q1_D_Can.png', dpi=400,bbox_inches='tight')
#plt.show()
plt.clf()


#------------Singapore: ----------------



# group by host and count the number of occurrences
host_counts_sg = hosts_sg.select('host').groupBy('host').count().sort(desc('count'))

# extract the most frequent host
most_frequent_host_sg = host_counts_sg.first()['host']

# filter the original DataFrame based on the most frequent host
most_frequent_host_visits_sg = hosts_sg.filter(hosts_sg['host'] == most_frequent_host_sg)

# extract the times and hours of the visits
times_and_hours_sg =most_frequent_host_visits_sg.select(substring('timestamp', 1, 2).alias('day'), substring('timestamp', 13, 2).alias('hour'))

date_hours_pd_sg = times_and_hours_sg.toPandas()



# count the number of visits on each date-hour combination
counts_sg = date_hours_pd_sg.groupby(['day', 'hour']).size().reset_index(name='count')

# create pivot table for heatmap
heatmap_data_sg = pd.pivot_table(counts_sg, values='count', index=['hour'], columns=['day'])

# plot heatmap
plt.pcolormesh(heatmap_data_sg, cmap='Oranges')
plt.colorbar()
plt.title(f'Singapore Top Host: {most_frequent_host_sg}')
plt.xlabel('Date')
plt.ylabel('Hour')
plt.xticks(np.arange(0.5, len(heatmap_data_sg.columns), 1), heatmap_data_sg.columns, rotation=90)
plt.yticks(np.arange(0.5, len(heatmap_data_sg.index), 1), heatmap_data_sg.index)
plt.savefig('Output/Q1_D_Sing.png', dpi=400,bbox_inches='tight')
#plt.show()
plt.clf()

#-----------------------------------------------------------------------------------------
