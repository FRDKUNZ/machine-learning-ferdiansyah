import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Informasi awal
print(df.info())
print(df.head())
print("Missing values:\n", df.isnull().sum())

# Hapus duplikat
df = df.drop_duplicates()

# Statistik deskriptif menggunakan NumPy
print("\nStatistik Deskriptif (NumPy):")
print("Rata-rata IPK:", np.mean(df['IPK']))
print("Median IPK:", np.median(df['IPK']))
print("Standar deviasi IPK:", np.std(df['IPK']))
print("Minimum IPK:", np.min(df['IPK']))
print("Maksimum IPK:", np.max(df['IPK']))

# Visualisasi EDA
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")

plt.subplot(2, 2, 2)
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")

plt.subplot(2, 2, 3)
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar")

plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi antar fitur")

plt.tight_layout()
plt.show()

# Fitur tambahan
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan dataset yang telah diproses
df.to_csv("processed_kelulusan.csv", index=False)

# Split data
X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nUkuran dataset:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)