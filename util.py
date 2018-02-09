def split_data(data, labels, proportion):

	size = data.shape[0]
	np.random.seed(42)
	s = np.random.permutation(size)
	split_idx = int(proportion * size)
	data = (data//255)
	labels2 = correctLabel(labels)
	return data[s[:split_idx]],
		   data[s[split_idx:]],
		   labels2[s[:split_idx]],
		   labels2[s[split_idx:]]

def correctLabel(labels):
	[n,m] = labels.shape
	if n == 0:
		temp = np.zeros(m,10)
		for i in m:
		 	labels2[i,labels[1,i]]=1
	else:
		temp = np.zeros(m,10)
		for i in n:
		 	labels2[i,labels[i,1]]=1
	
	return labels2
