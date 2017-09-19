import numpy as np

# X_i \sim N(1,1) i.i.d. -> generate \Sigma_X

def computeSigmaX(n):
	s = np.ones((n,n))
	for i in range(n):
		s[i,i] = 2
	return s

# X_i \sim N(1,1) i.i.d., Y indip, EY = c -> generate \Sigma_{YX}

def computeSigmaYX(c,n):
	s = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			s[i,j] = c[i]
	return s

# Delta[i,j,k]

def computeDelta(n):
	delta = np.zeros((n,n,n))
	for i in range(n):
		delta[i,i,i] = 1
	return delta

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
		
def computeSigmaX2(n):
	s = np.zeros((n,n,n,n))
	for i in range(n):
		for j in range(n):
			for k in range(n):
				for h in range(n): 
					s[i,j,k,h] = fourMomNorm([i,j,k,h])
	return s

# X_i \sim N(0,1) i.i.d., Y indip, EY = c -> generate \Sigma_{YX^{*2}}

def secMomNorm(v):
	u = np.unique(v)
	if len(u) == 2:
		return 0
	else:
		return 1

def computeSigmaYX2(c,n):
	m = len(c)
	s = np.zeros((m,n,n))
	for i in range(m):
		for j in range(n):
			for k in range(n):
				s[i,j] = c[i]*secMomNorm([k,j])
	return s

# X_i \sim N(0,1) i.i.d., Y = p(x,\theta^*) -> generate \Sigma_{YX^{*2}}
