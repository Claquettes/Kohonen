import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Créer le dossier de sauvegarde des images s'il n'existe pas
output_dir = 'results_plots'
os.makedirs(output_dir, exist_ok=True)

# Lire les trois fichiers CSV
df1 = pd.read_csv('results1.csv')
df2 = pd.read_csv('results2.csv')
df3 = pd.read_csv('results3.csv')

# Combiner les DataFrames pour faciliter l'analyse (optionnel)
df = pd.concat([df1, df2, df3], ignore_index=True)

# Afficher les premières lignes pour vérifier la structure
print(df.head())

# Exemple 1 : Erreur de Quantification vs Nombre d’Itérations (N)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='N', y='QuantificationError', hue='DatasetType', marker='o')
plt.title('Erreur de Quantification vs Nombre d’Itérations')
plt.xlabel('Nombre d’Itérations (N)')
plt.ylabel('Erreur de Quantification')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'quantification_vs_N.png'))
plt.show()

# Exemple 2 : Erreur d’Auto-Organisation vs Nombre d’Itérations (N)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='N', y='SelfOrganizationError', hue='DatasetType', marker='s')
plt.title('Erreur d’Auto-Organisation vs Nombre d’Itérations')
plt.xlabel('Nombre d’Itérations (N)')
plt.ylabel('Erreur d’Auto-Organisation')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'self_organization_vs_N.png'))
plt.show()

# Exemple 3 : Erreur de Quantification vs Eta (avec Sigma représenté par le style)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Eta', y='QuantificationError', hue='DatasetType', style='Sigma', s=100)
plt.title('Erreur de Quantification vs Eta (Sigma en style)')
plt.xlabel('Eta')
plt.ylabel('Erreur de Quantification')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'quantification_vs_Eta.png'))
plt.show()

# Exemple 4 : Erreur de Quantification vs Taille de la Grille (GridSize)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='GridSize', y='QuantificationError', hue='DatasetType')
plt.title('Erreur de Quantification vs Taille de la Grille')
plt.xlabel('Taille de la Grille (GridSize)')
plt.ylabel('Erreur de Quantification')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'quantification_vs_GridSize.png'))
plt.show()
