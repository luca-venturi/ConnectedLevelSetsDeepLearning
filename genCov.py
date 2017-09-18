import numpy as np

# X_i \sim N(0,1) i.i.d. -> generate \Sigma_{X^{*2}}

def fourMomNorm(v):
	u = np.unique(v)
	if len(u) > 2:
		return 0
	elif len(u) == 2:
		if max([v.count(u[k]) for k in range(2)]) == 2:
			return 1
		else:
			return 0
	elif len(u) == 1:
		return 3
		
def compute_sigmaX2(n):
	nsq = n**2
	s = np.zeros((nsq,nsq))
	for i in range(nsq):
		for j in range(nsq):
			s[i,j] = fourMomNorm([int(i/n),i%n,int(j/n),j%n])

	return s

# X_i \sim N(0,1) i.i.d., Y indip, EY = c -> generate \Sigma_{YX^{*2}}

def secMomNorm(v):
	u = np.unique(v)
	if len(u) == 2:
		return 0
	else:
		return 1

def compute_sigmaYX2(c,n):
	m = len(c)
	nsq = n**2
	s = np.zeros((m,nsq))
	for i in range(m):
		for j in range(nsq):
			s[i,j] = c[i]*secMomNorm([int(j/n),j%n])

	return s

# X_i \sim N(0,1) i.i.d., Y = p(x,\theta^*) -> generate \Sigma_{YX^{*2}}
