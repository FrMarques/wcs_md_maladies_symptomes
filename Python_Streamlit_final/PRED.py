import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Charger les fichiers CSV
df = pd.read_csv('ML_EDA.csv')
df2 = pd.read_csv('ML_EDA.csv')
precautions_df = pd.read_csv("Disease precaution.csv")

# Nettoyage des données (similaire à votre code)
df = df.drop(columns=['Disease'])
df.columns = df.columns.str.strip()

# Diviser les données en X (symptômes) et y (maladies)
X = df.drop(columns=['Disease_Encode'])
y = df['Disease_Encode']

# Diviser les données en ensemble d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer l'instance du classificateur KNN avec un nombre de voisins (par exemple, k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Liste des symptômes
symptoms_list = ['itching', 'skin_rash', 'continuous_sneezing', 'shivering', 'stomach_pain', 'acidity', 'vomiting',
                 'indigestion', 'muscle_wasting', 'patches_in_throat', 'fatigue', 'weight_loss', 'sunken_eyes', 'cough',
                 'headache', 'chest_pain', 'back_pain', 'weakness_in_limbs', 'chills', 'joint_pain', 'yellowish_skin',
                 'constipation', 'pain_during_bowel_movements', 'breathlessness', 'cramps', 'weight_gain', 'mood_swings',
                 'neck_pain', 'muscle_weakness', 'stiff_neck', 'pus_filled_pimples', 'burning_micturition', 'bladder_discomfort',
                 'high_fever', 'nodal_skin_eruptions', 'ulcers_on_tongue', 'loss_of_appetite', 'restlessness', 'dehydration',
                 'dizziness', 'weakness_of_one_body_side', 'lethargy', 'nausea', 'abdominal_pain', 'pain_in_anal_region', 'sweating',
                 'bruising', 'cold_hands_and_feets', 'anxiety', 'knee_pain', 'swelling_joints', 'blackheads', 'foul_smell_of urine',
                 'skin_peeling', 'blister', 'dischromic _patches', 'watering_from_eyes', 'extra_marital_contacts', 'diarrhoea',
                 'loss_of_balance', 'blurred_and_distorted_vision', 'altered_sensorium', 'dark_urine', 'swelling_of_stomach',
                 'bloody_stool', 'obesity', 'hip_joint_pain', 'movement_stiffness', 'spinning_movements', 'scurring',
                 'continuous_feel_of_urine', 'silver_like_dusting', 'red_sore_around_nose', 'Unnamed: 74', 'spotting_ urination',
                 'passage_of_gases', 'irregular_sugar_level', 'family_history', 'lack_of_concentration', 'excessive_hunger',
                 'yellowing_of_eyes', 'distention_of_abdomen', 'irritation_in_anus', 'swollen_legs', 'painful_walking', 'small_dents_in_nails',
                 'yellow_crust_ooze', 'internal_itching', 'mucoid_sputum', 'history_of_alcohol_consumption', 'swollen_blood_vessels',
                 'unsteadiness', 'inflammatory_nails', 'depression', 'fluid_overload', 'swelled_lymph_nodes', 'malaise', 'prominent_veins_on_calf',
                 'puffy_face_and_eyes', 'fast_heart_rate', 'irritability', 'muscle_pain', 'mild_fever', 'yellow_urine', 'phlegm', 'enlarged_thyroid',
                 'increased_appetite', 'visual_disturbances', 'brittle_nails', 'drying_and_tingling_lips', 'polyuria', 'pain_behind_the_eyes',
                 'toxic_look_(typhos)', 'throat_irritation', 'swollen_extremeties', 'slurred_speech', 'red_spots_over_body', 'belly_pain',
                 'receiving_blood_transfusion', 'acute_liver_failure', 'redness_of_eyes', 'rusty_sputum', 'abnormal_menstruation',
                 'receiving_unsterile_injections', 'coma', 'sinus_pressure', 'palpitations', 'stomach_bleeding', 'runny_nose', 'congestion', 'blood_in_sputum', 'loss_of_smell']

# Fonction pour transformer les symptômes de l'utilisateur en une ligne d'entrée pour le modèle
def transform_symptoms_to_features(user_symptoms, symptoms_list):
    features = [0] * len(symptoms_list)  # Crée un vecteur de zéros
    for symptom in user_symptoms:
        index = symptoms_list.index(symptom)
        features[index] = 1  # On indique que ce symptôme est présent
    return np.array(features).reshape(1, -1)

# Fonction pour afficher les précautions associées à une maladie
def afficher_precautions(maladie_predite, precautions_df):
    if maladie_predite in precautions_df['Disease'].values:
        prec_row = precautions_df[precautions_df['Disease'] == maladie_predite]
        precautions = prec_row.iloc[0, 1:]  # Extraire toutes les précautions
        precautions = precautions.dropna()  # Supprimer les NaN
        return precautions.tolist()
    else:
        return []

# Interface utilisateur Streamlit
st.title("Pré-Diagnostic Médical Virtuel")

# Permet à l'utilisateur de sélectionner les symptômes
selected_symptoms = st.multiselect("Veuillez sélectionnez 5 symptômes parmi la liste suivante :", symptoms_list, max_selections=5)

# Vérifier si l'utilisateur a sélectionné exactement 5 symptômes
if len(selected_symptoms) < 5:
    st.warning("Veuillez sélectionner exactement 5 symptômes pour activer le diagnostic.")
elif len(selected_symptoms) == 5:
    if st.button("Obtenir le Diagnostic"):
        # Transformer les symptômes en vecteur de caractéristiques
        user_features = transform_symptoms_to_features(selected_symptoms, symptoms_list)

        # Prédire la maladie
        predicted_disease_code = knn.predict(user_features)[0]
        maladie_predite = df2[df2['Disease_Encode'] == predicted_disease_code]['Disease'].iloc[0]
        
        # Afficher la maladie prédite
        st.subheader(f"Maladie prédite par le Pré-Diagnostic Médical Virtuel : {maladie_predite}")

        # Afficher les précautions associées
        precautions = afficher_precautions(maladie_predite, precautions_df)
        if precautions:
            st.write("Mesures préventives conseillées :")
            for i, precaution in enumerate(precautions, 1):
                st.write(f"{i}. {precaution}")
        else:
            st.write("Aucune précaution trouvée pour cette maladie.")
