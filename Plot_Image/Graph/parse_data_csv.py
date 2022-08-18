import re
import csv


fnm="cifar10_ica_norm_5-5"#input("input file name:")#'./merge_model_ppc4.txt'
print fnm
fname=""
fname="./" + str(fnm) + ".txt"
data=["Epoch",'loss','accuracy','val_loss','val_acc']
m=[]
epoch_counter=0

with open(fname) as f:
	content = f.readlines()
	#print content
	i=0
	csvfn=str(fnm) + ".csv"
	#m =re.split('[ - ]', str)
	c = csv.writer(open(csvfn, "wb"))
	c.writerow(data)
	i = 0
	'''for i in range (len(m)):
		temp=m[i]

		if(temp=="loss:"):
			temp=m[i]+m[i+1]

			print temp
		i=i+1'''
	for i in range (len(content)) :
		#print content[i]
		str=content[i]
		#print str
		m =re.split('[ - ]', str)
		#print m

		'''if (m[0]=="Epoch"):
			#data.append(m[1])'''

		for i in range (len(m)):
			temp=m[i]
			'''if (temp=="Epoch"):
				data=[]
				data.append(m[i+1])'''

			if(temp=="loss:"):
				epoch_counter=epoch_counter+1
				data=[]
				#data.append(epoch_counter)
				data.append(epoch_counter)
				data.append(m[i+1])
			if(temp=="acc:"):
				#data.append(m[i])
				data.append(m[i+1])
			if(temp=="val_loss:"):
				#data.append(m[i])
				data.append(m[i+1])
			if(temp=="val_acc:"):
				#data.append(m[i])
				print m[i+1]


				data.append(float(m[i+1]))
				print data
				c.writerow(data)
