# Εισαγωγή απαραίτητων βιβλιοθηκών
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Συνάρτηση για την προετοιμασία των δεδομένων
def prepare_data(df, train_size, shuffle, random_state):
    # Αφαίρεση των χαρακτηριστικών Month, Browser, OperatingSystems από το DataFrame
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # Μετατροπή των στηλών Revenue και Weekend σε ακέραιους (0 και 1)
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)
    
    # Εφαρμογή One-hot encoding στις κατηγορικές μεταβλητές Region, TrafficType, VisitorType
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # Διαχωρισμός των δεδομένων σε χαρακτηριστικά (X) και στόχο (y)
    X = df.drop(columns=['Revenue'])  # Η στήλη στόχος είναι το 'Revenue'
    y = df['Revenue']
    
    # Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle, random_state=random_state
    )
    
    # Επιστροφή των συνόλων εκπαίδευσης και δοκιμής
    return X_train, X_test, y_train, y_test

# Φόρτωση των δεδομένων από το αρχείο CSV
df = pd.read_csv('project2_dataset.csv')

# Προετοιμασία των δεδομένων χρησιμοποιώντας την συνάρτηση prepare_data
X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, shuffle=True, random_state=42)

# Δημιουργία του αντικειμένου MinMaxScaler για κανονικοποίηση των δεδομένων
scaler = MinMaxScaler()

# Υπολογισμός των παραμέτρων κανονικοποίησης στο σύνολο εκπαίδευσης
scaler.fit(X_train)

# Εφαρμογή του μετασχηματισμού στα σύνολα εκπαίδευσης και δοκιμής
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Δημιουργία και εκπαίδευση του Logistic Regression μοντέλου χωρίς κανονικοποίηση (penalty='none') και με μέγιστες επαναλήψεις 1000
model = LogisticRegression(penalty='none', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Προβλέψεις και πιθανότητες πρόβλεψης για το σύνολο εκπαίδευσης
y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
# Προβλέψεις και πιθανότητες πρόβλεψης για το σύνολο δοκιμής
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
# Προβλέψεις για το σύνολο εκπαίδευσης
y_train_pred = model.predict(X_train_scaled)
# Προβλέψεις για το σύνολο δοκιμής
y_test_pred = model.predict(X_test_scaled)

# Αξιολόγηση του μοντέλου με μέτρηση ακρίβειας για το σύνολο εκπαίδευσης
train_accuracy = accuracy_score(y_train, y_train_pred)
# Αξιολόγηση του μοντέλου με μέτρηση ακρίβειας για το σύνολο δοκιμής
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Ακρίβεια στο σύνολο εκπαίδευσης: ", train_accuracy)
print("Ακρίβεια στο σύνολο δοκιμής: ", test_accuracy)

# Υπολογισμός του πίνακα σύγχυσης για το σύνολο εκπαίδευσης
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
# Υπολογισμός του πίνακα σύγχυσης για το σύνολο δοκιμής
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Εκτύπωση του πίνακα σύγχυσης για το σύνολο εκπαίδευσης
print("Πίνακας σύγχυσης (Εκπαίδευση):\n", conf_matrix_train)
# Εκτύπωση του πίνακα σύγχυσης για το σύνολο δοκιμής
print("Πίνακας σύγχυσης (Δοκιμή):\n", conf_matrix_test)

# Δημιουργία γραφήματος για τον πίνακα σύγχυσης
plt.figure(figsize=(10, 4))
# Heatmap για τον πίνακα σύγχυσης στο σύνολο εκπαίδευσης
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Πίνακας Σύγχυσης (Εκπαίδευση)')
plt.xlabel('Προβλέψη (1=>αληθές ,0=>ψευδές)')
plt.ylabel('Πραγματικά (1=>αληθές ,0=>ψευδές)')

# Heatmap για τον πίνακα σύγχυσης στο σύνολο δοκιμής
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Πίνακας Σύγχυσης (Δοκιμή)')
plt.xlabel('Προβλέψη (1=>αληθές ,0=>ψευδές)')
plt.ylabel('Πραγματικά (1=>αληθές ,0=>ψευδές)')

# Ρύθμιση του διαγράμματος για καλή εμφάνιση
plt.tight_layout()
plt.show()
