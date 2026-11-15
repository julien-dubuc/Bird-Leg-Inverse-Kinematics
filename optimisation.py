import os, warnings
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize
from model import model

FICHIER = "./fich-from-data-scilab/joints_xyz_average_in_VCp_ref_frame_q12.xlsx"
BORNES = [(-180,180),(10,170),(-180,180),(10,170)]  # psi_h,eta_h,psi_a,eta_a
Q_MAX = np.radians(90)  # butée articulaire
P_INIT = [116,55,-93,67]  # angles du sujet
P_DROIT = [90,90,90,90]  # axes droits
warnings.filterwarnings("ignore",category=RuntimeWarning)  # axes dégénérés -> coût 1e3

# --- données, repère hanche, pose de repos ---
ws = openpyxl.load_workbook(FICHIER).active
data = np.array([[float(c.value) for c in row] for row in ws.iter_rows(2,ws.max_row)])
toe,foot,ankle,knee,hip = data[:,0:3],data[:,3:6],data[:,6:9],data[:,9:12],data[:,12:15]
N = len(foot)
Foot_mes,Ankle_mes,Knee_mes = foot-hip,ankle-hip,knee-hip
ecart = np.abs(knee[:,0]-(foot[:,0]+toe[:,0])/2)  # genou au milieu du pied-orteil
ind_start = int(np.argmin(ecart[:N-1]))  # on saute le dernier (modulo)
points = np.array([hip[ind_start],knee[ind_start],ankle[ind_start],foot[ind_start],toe[ind_start]])

# --- boucle de cinématique inverse ---
def cinematique_inverse(ang_hip,ang_ankle,nu=1.0):
    leg = model(points,np.array(ang_hip),np.array(ang_ankle))
    q = np.zeros(4)  # q=0 -> pose de repos
    Q = np.zeros((N,4)); Kq = np.zeros((N,3)); Aq = np.zeros((N,3)); Fq = np.zeros((N,3)); dF = np.zeros(N)
    Kq[ind_start],Aq[ind_start],Fq[ind_start] = Knee_mes[ind_start],Ankle_mes[ind_start],Foot_mes[ind_start]
    Fcur = Foot_mes[ind_start]
    for m in range(1,N+1):
        i = (m+ind_start)%N  # tour complet du cycle
        JK,JA,JF = leg.calc_JacobianKneeAnkleFoot(*q)
        Jp = LA.pinv(JF)  # 4x3
        q = q+Jp@(Foot_mes[i]-Fcur)-nu*(np.eye(4)-Jp@JF)@q  # écart à la position atteinte + rappel noyau
        K,A,F = leg.calc_KneeAnkleFootCoord(*q)
        Kq[i],Aq[i],Fq[i] = np.squeeze(K[:3]),np.squeeze(A[:3]),np.squeeze(F[:3])
        Fcur = Fq[i]; Q[i] = q; dF[i] = LA.norm(Foot_mes[i]-Fq[i])
    return Q,Kq,Aq,Fq,dF

# --- coût (= erreurs genou+cheville, pénalité butée) ---
def cout(p):
    try:
        Q,Kq,Aq,_,_ = cinematique_inverse(p[0:2],p[2:4])
    except Exception:
        return 1e3  # axes dégénérés
    c = (np.sqrt(np.mean(np.sum((Kq-Knee_mes)**2,axis=1)))  # RMS 3D genou
         +np.sqrt(np.mean(np.sum((Aq-Ankle_mes)**2,axis=1))))  # RMS 3D cheville
    if not np.isfinite(c): return 1e3  # sécurité NaN/inf
    d = np.max(np.abs(Q))-Q_MAX  # dépassement butée
    return c+10*d if d>0 else c

# --- optimisation multi-départ (le coût a des minima locaux) ---
departs = [P_INIT,[90,90,90,90],[0,45,0,45],[-90,120,90,60]]
res_list = [minimize(cout,p0,method="Nelder-Mead",bounds=BORNES,
                     options={"xatol":0.1,"fatol":1e-6,"maxiter":300}) for p0 in departs]  # pas de gradient
for p0,r in zip(departs,res_list):
    print(f"départ {p0} -> {np.round(r.x,1)}  coût={1000*r.fun:.2f}mm")
meilleur = min(res_list,key=lambda r:round(r.fun,4))  # a egalite (axes opposes) on garde le premier
P_OPT = meilleur.x
print(f"\nangles optimaux={np.round(P_OPT,1)}  coût={1000*meilleur.fun:.2f}mm")

# --- trois jeux et résumé ---
jeux = [("sujet",P_INIT,cinematique_inverse(P_INIT[:2],P_INIT[2:]),"tab:red"),
        ("axes droits",P_DROIT,cinematique_inverse(P_DROIT[:2],P_DROIT[2:]),"tab:purple"),
        ("optimisé",P_OPT,cinematique_inverse(P_OPT[:2],P_OPT[2:]),"tab:green")]
for nom,p,(Q,Kq,Aq,Fq,dF),c in jeux:
    ek = 1000*LA.norm(Kq-Knee_mes,axis=1); ea = 1000*LA.norm(Aq-Ankle_mes,axis=1)  # en mm
    print(f"{nom:12s} genou {np.sqrt(np.mean(ek**2)):.1f}/{ek.max():.1f}  "
          f"cheville {np.sqrt(np.mean(ea**2)):.1f}/{ea.max():.1f}  "
          f"pied {1000*dF.mean():.3f}/{1000*dF.max():.3f}  "
          f"J={1000*cout(p):.2f}  |q|max={np.round(np.degrees(np.abs(Q).max(axis=0)),1)}")

# --- figures ---
os.makedirs("figures",exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":110})
cycle = np.arange(N)
noms_q = [r"$q_{H1}$(roulis)",r"$q_{H2}$(hanche)",r"$q_K$(genou)",r"$q_A$(cheville)"]
couleurs = ["gray","tab:blue","tab:green","tab:orange"]

# fig2 : effet du terme de noyau
Q_nu0 = cinematique_inverse(P_INIT[:2],P_INIT[2:],nu=0.0)[0]  # sans rappel
fig,axs = plt.subplots(1,2,figsize=(10.5,3.8),sharey=True)
for ax,Qn,t in zip(axs,[Q_nu0,jeux[0][2][0]],[r"$\nu=0$",r"$\nu=1$"]):
    for j in range(4):
        ax.plot(cycle,np.degrees(Qn[:,j]),c=couleurs[j],label=noms_q[j])
    ax.axvline(ind_start,c="k",ls=":",lw=1); ax.set_xlabel("échantillon"); ax.set_title(t)  # pose de repos
axs[0].set_ylabel("angle(°)"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig2_noyau.png",bbox_inches="tight")

# fig3 : trajectoires vue de côté
fig,axs = plt.subplots(1,3,figsize=(13,3.8),sharey=True)
for ax,(nom,p,(Q,Kq,Aq,Fq,dF),c) in zip(axs,jeux):
    for M,lab,col in [(Knee_mes,"genou","tab:orange"),(Ankle_mes,"cheville","tab:green"),(Foot_mes,"pied","tab:blue")]:
        ax.plot(M[:,0],M[:,2],c=col,lw=3,alpha=0.5,label=lab+" mesuré")
    for X in (Kq,Aq,Fq): ax.plot(X[:,0],X[:,2],"r--",lw=1.2)  # modèle
    ax.plot([],[],"r--",label="modèle"); ax.set_aspect("equal")  # juste pour la légende
    ax.set_xlabel("X(m)"); ax.set_title(f"{nom}\n{tuple(round(v) for v in p)}")
axs[0].set_ylabel("Z(m)")
h,l = axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,fontsize=8)
plt.tight_layout(rect=(0,0.11,1,1)); plt.savefig("figures/fig3_trajectoires.png",bbox_inches="tight")

# fig4 : erreur point par point
fig,axs = plt.subplots(1,2,figsize=(10.5,3.8))
for ax,titre,ref,k in zip(axs,["Genou","Cheville"],[Knee_mes,Ankle_mes],[1,2]):
    for nom,p,res_,c in jeux:
        ax.plot(cycle,1000*LA.norm(res_[k]-ref,axis=1),c=c,label=nom)
    ax.set_title(titre); ax.set_xlabel("échantillon"); ax.set_ylabel("erreur(mm)")
axs[0].legend(); plt.tight_layout(); plt.savefig("figures/fig4_erreurs.png",bbox_inches="tight")

# fig5 : carte de coût, une paire varie, l'autre reste à l'optimum
psi_grid = np.arange(-180,181,10); eta_grid = np.arange(10,171,10)
cartes = [np.array([[cout([psi,eta]+list(P_OPT[2:4])) if k==0 else cout(list(P_OPT[0:2])+[psi,eta])
                     for psi in psi_grid] for eta in eta_grid]) for k in (0,1)]
fig,axs = plt.subplots(1,2,figsize=(12,4))
for ax,carte,pt,titre in zip(axs,cartes,[P_OPT[0:2],P_OPT[2:4]],
                             ["hanche, cheville fixée","cheville, hanche fixée"]):
    im = ax.pcolormesh(psi_grid,eta_grid,np.minimum(carte,0.03),shading="auto",cmap="viridis_r")  # coupe à 0.03 pour voir le creux
    ax.plot(pt[0],pt[1],"r*",ms=15,label="optimum")
    ax.plot((pt[0]+360)%360-180,180-pt[1],"o",mfc="none",mec="r",ms=11,mew=2,label="axe opposé")  # même axe, autre bout
    ax.set_xlabel(r"$\psi$(°)"); ax.set_ylabel(r"$\eta$(°)"); ax.set_title(titre)
    fig.colorbar(im,ax=ax,label="coût(m)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig5_cout.png",bbox_inches="tight")

# fig6 : cheville 3D
fig = plt.figure(figsize=(11,4.4))
for k,(nom,p,res_,c) in enumerate(jeux[1:]):
    ax = fig.add_subplot(1,2,k+1,projection="3d")
    A = 1000*res_[2]; M = 1000*Ankle_mes  # mm
    for i in range(0,N,2):  # un trait tous les 2 points pour ne pas surcharger
        ax.plot3D([M[i,0],A[i,0]],[M[i,1],A[i,1]],[M[i,2],A[i,2]],c="gray",lw=0.6)
    ax.plot3D(M[:,0],M[:,1],M[:,2],c="tab:green",lw=2.5,label="mesurée")
    ax.plot3D(A[:,0],A[:,1],A[:,2],"r--",lw=1.3,label="calculée")
    e = LA.norm(A-M,axis=1)
    ax.set_title(f"{nom}\nRMS {np.sqrt(np.mean(e**2)):.1f}mm, max {e.max():.1f}mm")
    ax.set_xlabel("X(mm)"); ax.set_ylabel("Y(mm)"); ax.set_zlabel("Z(mm)")
    ax.view_init(elev=22,azim=-60)
ax.legend(fontsize=8,loc="upper left")
plt.tight_layout(); plt.savefig("figures/fig6_cheville3d.png",bbox_inches="tight")

# fig7 : angles de commande
fig,axs = plt.subplots(1,4,figsize=(13,3.1))
for j,ax in enumerate(axs):
    for nom,p,res_,c in jeux[1:]:
        qj = np.degrees(res_[0][:,j])
        ax.plot(cycle,qj,c=c,label=nom); ax.axhline(qj.mean(),c=c,ls=":",lw=1)  # moyenne
    ax.set_title(noms_q[j],fontsize=9); ax.set_xlabel("échantillon")
axs[0].set_ylabel("angle(°)"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig7_angles.png",bbox_inches="tight")

# fig8 : écarts avec les angles du sujet
Q,Kq,Aq,Fq,dF = jeux[0][2]
fig,axs = plt.subplots(1,2,figsize=(10.5,3.1))
axs[0].plot(cycle,1000*LA.norm(Kq-Knee_mes,axis=1),c="tab:orange",label="genou")
axs[0].plot(cycle,1000*LA.norm(Aq-Ankle_mes,axis=1),c="tab:green",label="cheville")
axs[0].set_title("Genou, cheville (non commandés)"); axs[0].legend()
axs[1].plot(cycle,1000*dF,c="tab:blue"); axs[1].set_title("Pied (commandé)")
for ax in axs: ax.set_xlabel("échantillon"); ax.set_ylabel("écart(mm)")
plt.tight_layout(); plt.savefig("figures/fig8_ecarts_sujet.png",bbox_inches="tight")

plt.show()