import tensorflow as tf
import numpy as np

# problem size definitions
n = 3
m = 0
k = 7
cones = {'l':0, 'q': [3,4]}

# placeholder definitions
x = tf.placeholder(shape=(n), dtype=tf.float32, name="x")
y = tf.placeholder(shape=(m), dtype=tf.float32, name="y")
z = tf.placeholder(shape=(k), dtype=tf.float32, name="z")
s = tf.placeholder(shape=(k), dtype=tf.float32, name="s")

A = tf.placeholder(shape=(m, n), dtype=tf.float32, name="A") 
G = tf.placeholder(shape=(k, n), dtype=tf.float32, name="G") 
c = tf.placeholder(shape=(n), dtype=tf.float32, name="c")
b = tf.placeholder(shape=(m), dtype=tf.float32, name="b")
h = tf.placeholder(shape=(k), dtype=tf.float32, name="h")

# compute the degree of the cones
def conedeg(cones):
	return cones['l'] + len(cones['q'])

# make the identity element for the given product cone
def make_e(cones):
	arrs = []
	if cones['l'] > 0:
		arrs.append(np.ones(cones['l'], np.float32))
	for q in cones['q']:
		arrs.append([1])
		arrs.append(np.zeros(q-1, np.float32))
	return tf.constant(np.concatenate(arrs), dtype=tf.float32)

# compute the vector product of x and y wrt cones
def vprod(cones, x, y):
	outp = []
	if (cones['l'] > 0):
		end = cones['l']
		outp.append(x[0:end] * y[0:end])
	start = cones['l']
	for q in cones['q']:
		end = start + q
		e1 = tf.expand_dims(tf.tensordot(x[start:end], y[start:end], 1), 0)
		er = x[start] * y[start+1:end] + y[start] * x[start+1:end]
		outp.append(tf.concat([e1, er], 0))
		start += q
	return tf.concat(outp, 0)

# compute the inverse vector product of x and y wrt cones
def iprod(cones, x, y):
	outp = []
	if (cones['l'] > 0):
		end = cones['l']
		outp.append(y[0:end]/x[0:end])
	start = cones['l']
	for q in cones['q']:
		end = start + q
		xr = x[start+1:end]
		yr = y[start+1:end]
		xrem = tf.tensordot(xr, xr, 1)
		xsum = tf.reduce_sum(xr)
		xydp = tf.tensordot(xr, yr, 1)
		scf = 1.0/(x[start]**2 - xrem)
		e1 = tf.expand_dims(x[start] * y[start] - xydp, 0)
		er = -xr*y[start] + (x[start]**2 - xrem)/x[start]*yr + xr*xydp/x[start]
		outp.append(tf.concat([scf * e1,scf * er], 0))
		start += q
	return tf.expand_dims(tf.concat(outp, 0),1)

# compute the scaling matrices W and W^(-1) [denoted iW] for s z and wrt cones
def make_scaling(cones, s, z):
	w = tf.zeros((0,0),tf.float32)
	iw = tf.zeros((0,0),tf.float32)
	l = tf.zeros(0, tf.float32)

	if (cones['l'] > 0):
		end = cones['l']
		w = tf.diag(tf.sqrt(s[0:end] / z[0:end]))
		iw = tf.diag(tf.sqrt(z[0:end] / s[0:end]))
		l = tf.sqrt(s[0:end] * z[0:end])
	start = cones['l']
	for q in cones['q']:
		end = start + q
		zr = z[start:end]
		sr = s[start:end]
		nrmz = tf.sqrt(zr[0]**2 - tf.tensordot(zr[1:q],zr[1:q],1))
		nrms = tf.sqrt(sr[0]**2 - tf.tensordot(sr[1:q],sr[1:q],1))
		zb = zr/nrmz
		sb = sr/nrms
		gamma = tf.sqrt((1+tf.tensordot(zb, sb, 1))/2)
		wb1 = (sb[0] + zb[0])/(2*gamma)
		wbk = (sb[1:q] - zb[1:q])/(2*gamma)
		nrf = tf.sqrt(tf.sqrt((sr[0]*sr[0] - tf.tensordot(sr[1:q], sr[1:q], 1))/(zr[0]*zr[0] - tf.tensordot(zr[1:q], zr[1:q], 1))))

		core = tf.eye(q-1) + tf.matmul(tf.expand_dims(wbk,1), tf.expand_dims(wbk,1), transpose_b=True)/(wb1+1)
		r1 = tf.concat([tf.expand_dims(wb1, 0), wbk], 0)
		rk = tf.concat([tf.expand_dims(wbk, 1), core], 1)
		wk = nrf*tf.concat([tf.expand_dims(r1,0), rk], 0)

		r1i = tf.concat([tf.expand_dims(wb1, 0), -wbk], 0)
		rki = tf.concat([tf.expand_dims(-wbk, 1), core], 1)
		wki = tf.concat([tf.expand_dims(r1i, 0), rki], 0)/nrf

		lnrmf = tf.sqrt(nrmz*nrms)
		l0 = tf.expand_dims(gamma,0)
		lk = ((gamma + zb[0])*sb[1:q] + (gamma+sb[0])*zb[1:q])/(sb[0] + zb[0] + 2*gamma)

		#uhlc = start,start
		uppad = tf.zeros((start,q), tf.float32)
		lepad = tf.zeros((q,start), tf.float32)
		w = tf.concat([tf.concat([w,uppad],1),tf.concat([lepad,wk],1)],0)
		iw = tf.concat([tf.concat([iw,uppad],1),tf.concat([lepad,wki],1)],0)
		l = tf.concat([l,lnrmf*l0,lnrmf*lk], 0)
		start += q
	return w,iw,l

# solve the KKT system using a cholesky factorization
# does not handle the case where the upper left is singular
def kktsolve(cones, dx, dy, dz, ipr):
	t1 = tf.matmul(W, ipr, transpose_a=True)
	t2 = dz - t1

	p0 = tf.matmul(iW, iW, transpose_b=True)
	p1 = tf.matmul(G, p0, transpose_a=True)
	p3 = tf.matmul(p1, G)
	L = tf.cholesky(p3)
	I = tf.eye(3,3)
	Li = tf.linalg.triangular_solve(L, I, lower=True)
	iR = tf.matmul(Li, Li, transpose_a=True)
	siR = tf.matmul(A, iR)
	lhs = tf.matmul(siR, A, transpose_b=True)

	p2 = tf.matmul(p1, t2) + dx
	rhs = tf.matmul(siR, p2) - dy

	lhsc = tf.cholesky(lhs)
	ciy = tf.cholesky_solve(lhsc, rhs)
	cy = tf.add(ciy, tf.zeros((3)), name = "cy")
	cx = tf.matmul(iR, p2 - tf.matmul(A, ciy, transpose_a=True), name="cx")
	cz = tf.matmul(p0, tf.matmul(G, cx) - t2, name="cz")
	cs = tf.matmul(W, ipr - tf.matmul(W, cz), transpose_a=True, name="cs")
	return (cx, cy, cz, cs)

# calculate the current maximum step that can be taken with the given correction direction
def max_step(cones, li, xi):
	steps = []
	if cones['l'] > 0:
		steps.append(tf.expand_dims(tf.reduce_max(-xi[0:cones['l']] / li[0:cones['l']]),0))
	start = cones['l']
	for q in cones['q']:
		end = start + q
		a = 1/tf.sqrt(li[start]**2 - tf.reduce_sum(li[start+1:end]**2))
		r11 = li[start]*xi[start]*a 
		reg2 = xi[start+1:end]
		r12 = a * tf.tensordot(li[start+1:end], xi[start+1:end], 1)
		r1 = (r11 - r12)
		cst = (r1 + xi[start])/(li[start]*a + 1)
		r2 = a*(tf.squeeze(xi[start+1:end]) - cst * li[start+1:end]*a)
		r2n = tf.norm(r2)
		steps.append(r2n - r1*a)
		start += q
	return tf.reduce_max(tf.concat(steps, 0))

def compute_step(cones, l, dz, ds):
	mxz = max_step(cones, l, dz)
	mxs = max_step(cones, l, ds)
	t = tf.maximum(mxz, mxs)
	return tf.cond(t < 0.0, lambda: 1.0, lambda: tf.minimum(1.0, 1.0/t))

# example data
fd = { x: [-0.622877, -1.3237, 0.980248], 
 y: [], 
 z: [19.6436, -6.02755, 16.2913, 16.8376, 4.11306, 7.57566, 13.2861],
 s: [18.3706, 6.02755, -16.2913, 21.1766, -4.11306, -7.57566, -13.2861], A: np.zeros([0,3]),
 G: [[12.0,6.0,-5.0],[13.0,-3.0,-5.0],[12.0,-12.0,6.0],[3.0,-6.0,10.0],[3.0,-6.0,-2.0],[-1.0,-9.0,-2.0],[1.0,19.0,-3.0]],
 c: [-2.0, 1.0, 5.0], b:[], h:[-12.0, -3.0, -2.0, 27.0, 0.0, 3.0, -42.0]}

# single algorithm iteration
idel = make_e(cones)
W,iW,l = make_scaling(cones, s, z)
dx1 = tf.matmul(A, tf.expand_dims(y,1), transpose_a=True)
dx = dx1 + tf.matmul(G, tf.expand_dims(z,1), transpose_a=True) + tf.expand_dims(c, 1)
dy = tf.matmul(A, tf.expand_dims(x,1)) - tf.expand_dims(b,1)
dz = tf.matmul(G, tf.expand_dims(x,1)) + tf.expand_dims(s,1) - tf.expand_dims(h,1)
ds = vprod(cones, l, l)

ipr = iprod(cones, l, -ds)
cx, cy, cz, cs = kktsolve(cones, -dx, -dy, -dz, ipr)
t = compute_step(cones, l, tf.matmul(W, cz), tf.matmul(iW,cs, transpose_a=True))

lprod = tf.tensordot(l, l, 1)
rho = 1.0-t-t**2 * \
	tf.tensordot(tf.matmul(iW, cs, transpose_a=True), tf.matmul(W, cz), 2)/ \
	lprod
sig = tf.maximum(0.0, tf.minimum(1.0, rho))**3
mu = lprod/tf.constant(conedeg(cones), tf.float32)
 
scfact = 1.0-sig
ic1 = vprod(cones, tf.squeeze(tf.matmul(iW, cs, transpose_a=True)), tf.squeeze(tf.matmul(W, cz)))
comb_s = -ds + sig*mu*idel - ic1
rx,ry,rz,rs = kktsolve(cones, -dx*scfact, -dy*scfact, -dz*scfact, iprod(cones, l, comb_s))

alp = compute_step(cones, l, tf.matmul(W, rz), tf.matmul(iW,rs, transpose_a=True)) * 0.99

# outputs

nx = tf.add(tf.expand_dims(x,1), rx*alp, name="nx")
ny = tf.add(tf.expand_dims(y,1), ry*alp, name="ny")
nz = tf.add(tf.expand_dims(z,1), rz*alp, name="nz")
ns = tf.add(tf.expand_dims(s,1), rs*alp, name="ns")

tf.io.write_graph(tf.get_default_graph(), ".", "kktsolver.pbtxt")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print(sess.run([nx,ny,nz,ns], feed_dict=fd))