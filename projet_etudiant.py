# %% [markdown]
# PROJET : Analyse de données - régression linéaire, ANOVA et khi-2
# Nom, prénom : ................................    Groupe : ............
# Module : Analyse de Données et Classification
# Bibliothèques : pandas, numpy, scipy, matplotlib, seaborn
# Lancer ce fichier depuis le dossier du projet. Le CSV peut être dans data/ ou à la racine.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")
ALPHA = 0.05

# Les chemins sont relatifs au dossier où se trouve ce fichier.
DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_RESULTATS = DOSSIER_PROJET / "results"
DOSSIER_RESULTATS.mkdir(exist_ok=True)

chemins_csv = [
    DOSSIER_PROJET / "data" / "donnees_projet.csv",
    DOSSIER_PROJET / "donnees_projet.csv",
]
FICHIER_CSV = next((p for p in chemins_csv if p.exists()), None)

if FICHIER_CSV is None:
    raise FileNotFoundError(
        "Le fichier donnees_projet.csv est introuvable. "
        "Placez-le dans le dossier data/ ou à côté de ce script."
    )

df = pd.read_csv(FICHIER_CSV)
colonnes_attendues = [
    "id", "filiere", "sexe", "logement", "assiduite", "moyenne_bac",
    "heures_etude", "heures_sommeil", "note_stat", "reussite"
]
colonnes_absentes = [c for c in colonnes_attendues if c not in df.columns]
if colonnes_absentes:
    raise ValueError(f"Colonnes absentes du CSV : {colonnes_absentes}")

print("Fichier chargé :", FICHIER_CSV)
print("Dossier des résultats :", DOSSIER_RESULTATS)

# %% PARTIE 0 : chargement et nettoyage (2 pts)
print("\n========== PARTIE 0 : CHARGEMENT ET NETTOYAGE ==========")

# 0.1 : dimensions, types et valeurs manquantes
print("\n0.1 - Dimensions du dataset :", df.shape)
print("\nTypes des variables :")
print(df.dtypes)
print("\nValeurs manquantes par variable :")
print(df.isna().sum())
print("\nAperçu des cinq premières lignes :")
print(df.head())

# 0.2 : recherche des valeurs inhabituelles dans heures_etude
print("\n0.2 - Statistiques de heures_etude :")
print(df["heures_etude"].describe())

plt.figure(figsize=(7, 3.8))
sns.boxplot(x=df["heures_etude"], color="lightsteelblue")
plt.title("Boxplot des heures d'étude avant nettoyage")
plt.xlabel("Heures d'étude par semaine")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_00_boxplot_heures_etude.png", dpi=160)
plt.close()

Q1 = df["heures_etude"].quantile(0.25)
Q3 = df["heures_etude"].quantile(0.75)
IQR = Q3 - Q1
borne_inf = Q1 - 1.5 * IQR
borne_sup = Q3 + 1.5 * IQR

masque_outliers = (df["heures_etude"] < borne_inf) | (df["heures_etude"] > borne_sup)
outliers = df.loc[masque_outliers, ["id", "heures_etude", "note_stat"]]
print(f"\nQ1 = {Q1:.2f} ; Q3 = {Q3:.2f} ; IQR = {IQR:.2f}")
print(f"Bornes IQR : [{borne_inf:.3f} ; {borne_sup:.3f}]")
print("Observations détectées par la règle IQR :")
print(outliers.to_string(index=False))

# Le tableau original df reste inchangé. Dans ce jeu, l'observation id=18 vaut 95 h.
# Cette valeur est considérée comme une erreur de saisie probable et devient manquante
# dans la copie d'analyse. Les cinq autres valeurs élevées sont conservées après examen.
d = df.copy()
masque_95 = (d["id"] == 18) & (d["heures_etude"] == 95)
if masque_95.any():
    d.loc[masque_95, "heures_etude"] = np.nan
else:
    print("Attention : l'observation id=18 avec 95 h n'a pas été trouvée ; vérifier le traitement.")

print("\n0.3 - Tableau d'analyse :")
print("Nombre de lignes dans le fichier original :", len(df))
print("Nombre de lignes dans la copie d'analyse :", len(d))
print("Valeurs manquantes dans heures_etude après traitement :", d["heures_etude"].isna().sum())

# Pour la régression, seules les lignes avec les variables du modèle sont nécessaires.
d_reg = d.dropna(subset=["heures_etude", "note_stat", "moyenne_bac"]).copy()
print("Étudiants utilisés pour la régression :", len(d_reg))
print("Les autres analyses utilisent les 300 étudiants, car note_stat et les variables qualitatives sont complètes.")

# %% PARTIE 1 : analyse descriptive (2 pts)
print("\n========== PARTIE 1 : ANALYSE DESCRIPTIVE ==========")

# 1.1 : moyenne, écart-type, minimum et maximum
print("\n1.1 - Statistiques descriptives :")
stat_desc = d[["moyenne_bac", "heures_etude", "note_stat"]].agg(
    ["count", "mean", "std", "min", "max"]
).T
print(stat_desc.round(2))

# 1.2 : moyenne de note_stat par filière et effectifs qualitatifs
print("\n1.2 - Note en statistique par filière :")
print(d.groupby("filiere", observed=False)["note_stat"].agg(
    ["count", "mean", "median", "std"]
).round(2))

print("\nEffectifs des variables qualitatives :")
for variable in ["filiere", "sexe", "logement", "assiduite", "reussite"]:
    print(f"\n{variable} :")
    print(d[variable].value_counts())

# 1.3 : graphiques
plt.figure(figsize=(7, 4.2))
sns.histplot(d["note_stat"], bins=15, color="steelblue", edgecolor="white")
plt.title("Distribution de la note en statistique")
plt.xlabel("Note finale sur 20")
plt.ylabel("Nombre d'étudiants")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_1_histogramme_note.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4.2))
sns.boxplot(data=d, x="filiere", y="note_stat", color="lightsteelblue")
plt.title("Note en statistique selon la filière")
plt.xlabel("Filière")
plt.ylabel("Note finale sur 20")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_1_boxplot_note_filiere.png", dpi=160)
plt.close()

# %% PARTIE 2 : régression linéaire simple (6 pts)
print("\n========== PARTIE 2 : REGRESSION LINEAIRE ==========")
x = d_reg["heures_etude"].to_numpy()
y = d_reg["note_stat"].to_numpy()
n_reg = len(d_reg)

# 2.1 : nuage de points
plt.figure(figsize=(7, 4.5))
sns.scatterplot(x=x, y=y, alpha=0.65)
plt.title("Note en statistique selon les heures d'étude")
plt.xlabel("Heures d'étude par semaine")
plt.ylabel("Note sur 20")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_2_1_nuage_points.png", dpi=160)
plt.close()

# 2.2 : corrélation de Pearson
r, p_pearson = stats.pearsonr(x, y)
print("\n2.2 - Corrélation de Pearson")
print("H0 : rho = 0 ; H1 : rho est différent de 0")
print(f"r = {r:.4f} ; p-value = {p_pearson:.3e}")
print("Décision :", "rejet de H0" if p_pearson < ALPHA else "H0 non rejetée")

# 2.3 et 2.4 : droite, R², test de pente et intervalle de confiance
res = stats.linregress(x, y)
R2 = res.rvalue ** 2
ddl_reg = n_reg - 2
t_crit = stats.t.ppf(1 - ALPHA / 2, ddl_reg)
ic_bas = res.slope - t_crit * res.stderr
ic_haut = res.slope + t_crit * res.stderr
print("\n2.3 - Droite estimée")
print("H0 : beta1 = 0 ; H1 : beta1 est différent de 0")
print(f"Pente = {res.slope:.4f} ; ordonnée = {res.intercept:.4f}")
print(f"Equation : note = {res.intercept:.3f} + {res.slope:.3f} x heures_etude")
print("\n2.4 - Qualité de la régression")
print(f"R² = {R2:.4f} ; p-value de la pente = {res.pvalue:.3e}")
print(f"Erreur-type de la pente = {res.stderr:.4f}")
print(f"IC à 95 % de la pente = [{ic_bas:.4f} ; {ic_haut:.4f}]")

# 2.5 : calcul manuel de la pente et de l'ordonnée
x_moy = x.mean()
y_moy = y.mean()
pente_main = np.sum((x - x_moy) * (y - y_moy)) / np.sum((x - x_moy) ** 2)
ordonnee_main = y_moy - pente_main * x_moy
print("\n2.5 - Calcul manuel")
print(f"Pente = {pente_main:.4f} ; ordonnée = {ordonnee_main:.4f}")
print(f"Ecart de pente = {abs(pente_main - res.slope):.2e}")
print(f"Ecart d'ordonnée = {abs(ordonnee_main - res.intercept):.2e}")

# Graphique de la droite
plt.figure(figsize=(7, 4.5))
sns.scatterplot(x=x, y=y, alpha=0.60, label="Observations")
xs = np.linspace(x.min(), x.max(), 100)
plt.plot(xs, res.intercept + res.slope * xs, color="red", label="Droite estimée")
plt.title("Droite de régression : note en fonction des heures d'étude")
plt.xlabel("Heures d'étude par semaine")
plt.ylabel("Note sur 20")
plt.legend()
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_2_3_droite_regression.png", dpi=160)
plt.close()

# 2.6 : prédictions
print("\n2.6 - Prédictions")
print(f"Intervalle observé : [{x.min():.1f} ; {x.max():.1f}] heures")
for heures in [10, 40]:
    prediction = res.intercept + res.slope * heures
    print(f"Pour {heures} h : note prédite = {prediction:.2f}/20")

# 2.7 : résidus et vérification des conditions
valeurs_ajustees = res.intercept + res.slope * x
residus = y - valeurs_ajustees

plt.figure(figsize=(7, 4.2))
plt.scatter(valeurs_ajustees, residus, alpha=0.65)
plt.axhline(0, color="red", linestyle="--")
plt.title("Résidus en fonction des valeurs ajustées")
plt.xlabel("Valeurs ajustées")
plt.ylabel("Résidus")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_2_7_residus.png", dpi=160)
plt.close()

plt.figure(figsize=(5.5, 5.5))
stats.probplot(residus, dist="norm", plot=plt)
plt.title("QQ-plot des résidus")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "fig_2_7_qqplot.png", dpi=160)
plt.close()

W_res, p_res = stats.shapiro(residus)
mediane_ajustees = np.median(valeurs_ajustees)
ecart_faible = np.std(residus[valeurs_ajustees < mediane_ajustees], ddof=1)
ecart_fort = np.std(residus[valeurs_ajustees >= mediane_ajustees], ddof=1)
print("\n2.7 - Résidus")
print("H0 Shapiro : les résidus suivent une loi normale ; H1 : ils ne la suivent pas")
print(f"Shapiro-Wilk : W = {W_res:.4f} ; p-value = {p_res:.4f}")
print(f"Ecart-type des résidus (ajustées faibles) = {ecart_faible:.2f}")
print(f"Ecart-type des résidus (ajustées fortes) = {ecart_fort:.2f}")

# 2.8 : régression multiple (bonus) avec scikit-learn
modele_multiple = LinearRegression()
X_multiple = d_reg[["heures_etude", "moyenne_bac"]]
modele_multiple.fit(X_multiple, y)
R2_multiple = modele_multiple.score(X_multiple, y)
R2_ajuste = 1 - (1 - R2_multiple) * (n_reg - 1) / (n_reg - 3)
modele_bac_seul = LinearRegression().fit(d_reg[["moyenne_bac"]], y)
R2_bac_seul = modele_bac_seul.score(d_reg[["moyenne_bac"]], y)
print("\n2.8 - Régression multiple (bonus, scikit-learn)")
print(f"note = {modele_multiple.intercept_:.3f} + {modele_multiple.coef_[0]:.3f} x heures_etude + {modele_multiple.coef_[1]:.3f} x moyenne_bac")
print(f"R² heures seules = {R2:.4f} ; R² bac seul = {R2_bac_seul:.4f}")
print(f"R² multiple = {R2_multiple:.4f} ; R² ajusté = {R2_ajuste:.4f}")

# %% PARTIE 3 : ANOVA à un facteur (5 pts)
print("\n========== PARTIE 3 : ANOVA ==========")
ordre_filieres = ["Biologie", "Chimie", "Informatique"]
groupes = [d.loc[d["filiere"] == nom, "note_stat"].to_numpy() for nom in ordre_filieres]

# 3.1 : hypothèses et boxplot
print("H0 : les trois moyennes sont égales.")
print("H1 : au moins une moyenne est différente.")

# 3.2 : normalité par groupe et égalité des variances
print("\n3.2 - Vérification des conditions")
for nom, groupe in zip(ordre_filieres, groupes):
    W_g, p_g = stats.shapiro(groupe)
    print(f"Shapiro {nom} : W = {W_g:.4f} ; p = {p_g:.4f}")
stat_levene, p_levene = stats.levene(*groupes)
print(f"Levene : statistique = {stat_levene:.4f} ; p-value = {p_levene:.4f}")

# 3.3 : calcul manuel
N = len(d)
k = len(groupes)
moyenne_generale = d["note_stat"].mean()
SC_entre = sum(len(g) * (g.mean() - moyenne_generale) ** 2 for g in groupes)
SC_intra = sum(np.sum((g - g.mean()) ** 2) for g in groupes)
SC_total = SC_entre + SC_intra
ddl_entre = k - 1
ddl_intra = N - k
CM_entre = SC_entre / ddl_entre
CM_intra = SC_intra / ddl_intra
F_manuel = CM_entre / CM_intra
p_anova_manuel = stats.f.sf(F_manuel, ddl_entre, ddl_intra)
F_scipy, p_scipy = stats.f_oneway(*groupes)
print("\n3.3 - Calcul manuel et comparaison SciPy")
print(f"SC entre = {SC_entre:.4f} ; SC intra = {SC_intra:.4f} ; SC total = {SC_total:.4f}")
print(f"ddl entre = {ddl_entre} ; ddl intra = {ddl_intra}")
print(f"CM entre = {CM_entre:.4f} ; CM intra = {CM_intra:.4f}")
print(f"F manuel = {F_manuel:.4f} ; p-value = {p_anova_manuel:.3e}")
print(f"f_oneway : F = {F_scipy:.4f} ; p-value = {p_scipy:.3e}")

# 3.4 : valeur critique et décisions
F_critique = stats.f.ppf(1 - ALPHA, ddl_entre, ddl_intra)
print("\n3.4 - F critique =", round(F_critique, 4))
print("Décision avec F :", "rejet de H0" if F_manuel > F_critique else "H0 non rejetée")
print("Décision avec p-value :", "rejet de H0" if p_anova_manuel < ALPHA else "H0 non rejetée")

# 3.5 : taille d'effet
eta2 = SC_entre / SC_total
print(f"\n3.5 - Eta carré = {eta2:.4f}")

# 3.6 : comparaisons par paires de Tukey
print("\n3.6 - Comparaisons de Tukey")
tukey = stats.tukey_hsd(*groupes)
intervalle_tukey = tukey.confidence_interval()
for i in range(len(ordre_filieres)):
    for j in range(i + 1, len(ordre_filieres)):
        print(
            f"{ordre_filieres[i]} - {ordre_filieres[j]} : "
            f"différence = {tukey.statistic[i, j]:.3f} ; "
            f"p = {tukey.pvalue[i, j]:.3e} ; "
            f"IC 95 % = [{intervalle_tukey.low[i, j]:.3f} ; {intervalle_tukey.high[i, j]:.3f}]"
        )

# 3.7 : Kruskal-Wallis (bonus)
H_kw, p_kw = stats.kruskal(*groupes)
print("\n3.7 - Test de Kruskal-Wallis")
print("H0 : les distributions des notes sont identiques entre les filières.")
print(f"H = {H_kw:.4f} ; p-value = {p_kw:.3e}")
print("Décision :", "rejet de H0" if p_kw < ALPHA else "H0 non rejetée")

# %% PARTIE 4 : Khi-2 d'indépendance (5 pts)
print("\n========== PARTIE 4 : KHI-2 ==========")
# Ordre des modalités utilisé dans les tableaux et les graphiques.
d_chi = d.copy()
d_chi["logement"] = pd.Categorical(
    d_chi["logement"], categories=["Famille", "Cité universitaire", "Location"], ordered=True
)
d_chi["assiduite"] = pd.Categorical(
    d_chi["assiduite"], categories=["Faible", "Moyenne", "Forte"], ordered=True
)
d_chi["filiere"] = pd.Categorical(
    d_chi["filiere"], categories=["Biologie", "Chimie", "Informatique"], ordered=True
)
d_chi["sexe"] = pd.Categorical(d_chi["sexe"], categories=["F", "H"], ordered=True)

for var1, var2 in [("logement", "assiduite"), ("sexe", "filiere")]:
    print(f"\n--- {var1} x {var2} ---")
    print("H0 : les deux variables sont indépendantes.")
    print("H1 : les deux variables sont associées.")

    # 4.1 : tableau observé
    observe = pd.crosstab(d_chi[var1], d_chi[var2])
    print("\nEffectifs observés :")
    print(observe)

    # 4.2 : effectifs théoriques
    total = observe.to_numpy().sum()
    totaux_lignes = observe.sum(axis=1).to_numpy().reshape(-1, 1)
    totaux_colonnes = observe.sum(axis=0).to_numpy().reshape(1, -1)
    theorique = np.outer(totaux_lignes.ravel(), totaux_colonnes.ravel()) / total
    theorique = pd.DataFrame(theorique, index=observe.index, columns=observe.columns)
    print("\nEffectifs théoriques :")
    print(theorique.round(2))

    # 4.3 : contributions de chaque cellule
    contributions = (observe - theorique) ** 2 / theorique
    chi2_manuel = contributions.to_numpy().sum()
    print("\nContributions (O-E)²/E :")
    print(contributions.round(3))

    # 4.4 : ddl, valeur critique et comparaison avec SciPy
    ddl = (observe.shape[0] - 1) * (observe.shape[1] - 1)
    chi2_critique = stats.chi2.ppf(1 - ALPHA, ddl)
    p_khi2_manuel = stats.chi2.sf(chi2_manuel, ddl)
    resultat_chi2 = stats.chi2_contingency(observe, correction=False)
    print(f"Khi-2 manuel = {chi2_manuel:.4f} ; ddl = {ddl}")
    print(f"Khi-2 critique = {chi2_critique:.4f} ; p-value manuelle = {p_khi2_manuel:.4e}")
    print(f"SciPy : khi-2 = {resultat_chi2.statistic:.4f} ; ddl = {resultat_chi2.dof} ; p = {resultat_chi2.pvalue:.4e}")

    # 4.5 : condition des effectifs théoriques et décision
    minimum_theorique = theorique.to_numpy().min()
    print(f"Effectif théorique minimum = {minimum_theorique:.2f}")
    print("Condition E >= 5 :", "respectée" if minimum_theorique >= 5 else "non respectée")
    print("Décision :", "rejet de H0" if p_khi2_manuel < ALPHA else "H0 non rejetée")

    # 4.6 : V de Cramér et graphique en pourcentages
    V_cramer = np.sqrt(chi2_manuel / (total * (min(observe.shape) - 1)))
    print(f"V de Cramér = {V_cramer:.4f}")
    pourcentages = observe.div(observe.sum(axis=1), axis=0) * 100
    pourcentages.plot(kind="bar", stacked=True, figsize=(7, 4.5), colormap="Set2")
    plt.title(f"Répartition de {var2} selon {var1}")
    plt.xlabel(var1)
    plt.ylabel("Pourcentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title=var2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(DOSSIER_RESULTATS / f"fig_4_{var1}_{var2}.png", dpi=160)
    plt.close()

print("\nInterprétation : un khi-2 significatif indique une association statistique, mais ne prouve pas une relation de cause à effet.")

# %% PARTIE 5 : synthèse
print("\n========== PARTIE 5 : SYNTHESE ==========")
print(f"La régression indique une association positive entre heures d'étude et note (r = {r:.3f}, R² = {R2:.3f}).")
print(f"L'ANOVA donne F = {F_manuel:.2f}, p = {p_anova_manuel:.3e} et eta² = {eta2:.3f}.")
print(f"Kruskal-Wallis donne H = {H_kw:.2f}, p = {p_kw:.3e}.")
print("Le test logement-assiduité met en évidence une association statistique.")
print("Le test sexe-filière ne met pas en évidence d'association significative.")
print("Ces résultats décrivent des liens statistiques et ne permettent pas de conclure à une causalité.")
print("Les conclusions concernent les données simulées de cet exercice.")
print("\nAnalyse terminée. Les graphiques sont disponibles dans le dossier results.")