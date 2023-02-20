import os,time  # os library for manipulating file time and time library to calculate bandwidth
channel_file = '/file.txt'
interval = 0.005 # for synchronization
message = 0b10101001110110011010
msg_bin = bin(message) #converting to a binary string
msg = msg_bin[2:] #removing 0b from start
msg = msg.zfill(32) # padding to make it 32 bit long so as to fill one field of file time 
print('creating File\n')
cc = open(channel_file,'w')
cc.close()
print('\nSending message')
os.utime(channel_file,(int(msg, 2),0)) # setting file access time to the secret message padded with zeroes
time.sleep(1) # to allow receiver to catch up
os.remove(channel_file) # removing the channel file

# For Bandwidth Calculation
band_file = '/band.txt'
bf = open(band_file,'w')
bf.close()

ctr = 0

while ctr < 100 :
    r1 = 4294967295 # 32 -bit integer
    r2 = 4294967295 # 32 -bit integer
    os.utime(band_file,(r1, r2))
    time.sleep(interval)
    ctr += 1
time.sleep(interval) # allow receiver to catch up
os.remove(band_file)
print('Complete Exiting Now')