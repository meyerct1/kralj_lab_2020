
ld = [1,1,1,5,5,5,10,10,10,15,15,15,30,30,30,60,60,60]
tt = [10,15,30,15,30,60,30,60,90,60,90,120,90,120,180,120,180,240]
with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for ch in ['bf','gfp']:
	for l,t in zip(ld,tt):
		with open('commands.sh','a') as f:
			f.write('python unet_training_05082020_timesweep.py ' + str(l) + ' ' + str(t) + ' ' +  ch + '\n')
			f.write("echo 'Finished " + str(cnt) + "'\n")
			cnt = cnt + 1

