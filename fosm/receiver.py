import os,time
channel_file = '/file.txt'
band_file = '/band.txt' # to calculate bandwidth
interval = 0.005 
print("....Waiting For Message:\n")
while os.path.isfile(channel_file) == False: # receiver process waits till file is created
	pass

time.sleep(interval) # allows sender.py to set file time
received = ''

atime = bin(int(os.path.getatime(channel_file))) #accesses file access time which is a float value and converts it to binary string
atime = atime[2:] #removes 0b from start
received = received +atime   
time.sleep(interval)
msg_int = int(received) #converts str to int
print("Received message is : ",msg_int)


#Calculating Bandwidth over a longer message :
print('Calculating Bandwith over a longer data transmission')
while os.path.isfile(band_file) == False:
	pass

bnd_start = time.time()
msg = ''
try:
    while os.path.isfile(band_file) == True:
        access_time = bin(int(os.path.getatime(band_file)))
        ac = access_time[2:]
        msg = msg + ac
    
        
        modified_time = bin(int(os.path.getmtime(band_file)))
        mt = modified_time[2:]

        msg = msg + mt
        
      
        time.sleep(interval)
except FileNotFoundError:
    pass
bnd_end = time.time()
time_bnd = bnd_end - bnd_start
datalen = len(msg)
print("Time taken for transmission : ",time_bnd)
print("Number of bits transmitted : ",datalen)
print('Bandwidth of Covert Channel is : ' + str(datalen/ time_bnd) + ' '+' bits per second')

