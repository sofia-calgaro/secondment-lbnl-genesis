import numpy as np, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# real V00101Z profile (r,z) mm, from the GDML
P = np.array([[0,0],[12,0],[12,2],[15,2],[15,0],[41.4,0],[44.4,3],[44.4,87],[41.4,90],
              [7.93698985866658,90],[4,45],[4,37],[0,37]],float)
poly = Path(np.vstack([P,P[0]]))
def in_ge(p):
    return poly.contains_points(np.c_[np.hypot(p[:,0],p[:,1]), p[:,2]])

V, INS = 10.0, 0.05
# axis position inside the 10 mm grid is unknown here: scan offsets, pick a case with 7 Ge corners
best=None
rng=np.random.default_rng(0)
for ox in np.arange(0,10,1.0):
  for oy in np.arange(0,10,1.0):
    for iz in range(-1,10):
      for ix in range(-3,3):
        for iy in range(-3,3):
          lo=np.array([ix*V-ox, iy*V-oy, iz*V])
          t=np.array(list(itertools.product([INS,V-INS],repeat=3)))
          c=lo+t; g=in_ge(c)
          if g.sum()!=7: continue
          pts=lo+rng.random((4000,3))*V
          fr=in_ge(pts).mean()
          # want voxel straddling the bore cone/wall, with sizeable LAr part
          if np.hypot(*(lo[:2]+V/2))<9 and 38<lo[2]<90:
            if best is None or (1-fr)>best[0]:
              best=(1-fr,lo.copy(),c,g)
lar_frac,lo,corners,g=best
print("LAr fraction of voxel: %.1f%%, Ge corners: %d/8, lower corner (local frame): %s"%(100*lar_frac,g.sum(),lo))

fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection="3d")
# crystal revolved, 270 deg so the bore is visible
ph=np.linspace(0,1.5*np.pi,60)
rr,zz=P[:,0],P[:,1]
for i in range(len(P)):
    j=(i+1)%len(P)
    if rr[i]<1e-9 and rr[j]<1e-9: continue
    quads=[]
    for a,b in zip(ph[:-1],ph[1:]):
        quads.append([[rr[i]*np.cos(a),rr[i]*np.sin(a),zz[i]],[rr[j]*np.cos(a),rr[j]*np.sin(a),zz[j]],
                      [rr[j]*np.cos(b),rr[j]*np.sin(b),zz[j]],[rr[i]*np.cos(b),rr[i]*np.sin(b),zz[i]]])
    ax.add_collection3d(Poly3DCollection(quads,facecolor="0.6",edgecolor="none",alpha=0.35))
# cut-face outline (profile at phi=0 and 1.5pi)
for a in (0,1.5*np.pi):
    ax.plot(np.r_[P[:,0],P[0,0]]*np.cos(a),np.r_[P[:,0],P[0,0]]*np.sin(a),np.r_[P[:,1],P[0,1]],"k-",lw=1)
# voxel
x0,y0,z0=lo; e=[(0,V)]*3
cs=np.array(list(itertools.product([0,V],repeat=3)))+lo
edges=[(a,b) for a,b in itertools.combinations(range(8),2) if np.sum(np.abs(cs[a]-cs[b])>1e-9)==1]
ax.add_collection3d(Line3DCollection([(cs[a],cs[b]) for a,b in edges],colors="C1",linewidths=2))
cin=in_ge(cs+np.sign(lo*0+ (np.array([1,1,1])*0)) )  # exact corners may sit on boundary; use inset corners for colour
col=np.array(list(itertools.product([INS,V-INS],repeat=3)))+lo
gg=in_ge(col)
ax.scatter(*col[gg].T,c="red",s=60,label="corner in Ge (%d)"%gg.sum(),depthshade=False)
ax.scatter(*col[~gg].T,c="deepskyblue",s=90,edgecolors="k",label="corner in LAr (%d)"%(~gg).sum(),depthshade=False)
ax.set_box_aspect((1,1,1)); L=48
ax.set_xlim(-L,L);ax.set_ylim(-L,L);ax.set_zlim(0,95)
ax.set_box_aspect((2*L,2*L,95))
ax.set_xlabel("x [mm]");ax.set_ylabel("y [mm]");ax.set_zlabel("z [mm]")
ax.set_title("V00101Z (270° cut) with a 10 mm voxel that survives the 8-corner rule\n"
             "voxel is %.0f%% LAr / %.0f%% Ge"%(100*lar_frac,100*(1-lar_frac)))
ax.legend(loc="upper left"); ax.view_init(24,-55)
fig.savefig("voxel_borehole.png",dpi=180,bbox_inches="tight")
