import os
import sys
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# Configure Python paths for Spark
spark_home = os.environ.get('SPARK_HOME', '/opt/bitnami/spark')
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.9.5-src.zip'))

st.set_page_config(layout="wide")
tab_names = ["Model", "Visualization"]
tabs = st.tabs(tab_names)

with tabs[0]:
    # Initialize Spark session with error handling
    @st.cache_resource
    def init_spark():
        try:
            return SparkSession.builder \
                .appName("DiseasePredictionDashboard") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
        except Exception as e:
            st.error(f"Failed to initialize Spark: {str(e)}")
            st.stop()

    spark = init_spark()

    # Load model with verification
    @st.cache_resource
    def load_model():
        model_paths = [
            "/opt/bitnami/spark/rf_pipeline_model",
            "/opt/spark/rf_pipeline_model",
            "/tmp/rf_pipeline_model"
        ]
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = PipelineModel.load(path)
                    # st.success(f"Model loaded successfully from {path}")
                    return model
            except Exception as e:
                st.warning(f"Attempt failed for {path}: {str(e)}")
        
        st.error("Model not found in any standard locations. Please verify model path.")
        st.stop()

    model = load_model()

    # Full list of symptoms (132 total)
    symptoms = [
        "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering","chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting","vomiting","burning_micturition","spotting_ urination","fatigue","weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness","lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of urine","continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)","depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic _patches","watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","blood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"
    ]

    # UI Components
    st.title("Disease Prediction Dashboard")
    st.markdown("Select your symptoms and click **Predict**")

    # Create checkboxes in 6 columns
    input_data = {}
    cols = st.columns(5)
    for i, symptom in enumerate(symptoms):
        with cols[i % 5]:
            input_data[symptom] = st.checkbox(
                symptom.replace("_", " ").title(),
                key=symptom,
                help=f"Select if experiencing {symptom.replace('_', ' ')}"
            )

    # Count how many symptoms the user selected
    selected_symptoms = [s for s in symptoms if input_data[s]]
    symptom_count = len(selected_symptoms)

    # Disable the Predict button unless at least 3 symptoms are selected
    if symptom_count < 3:
        st.warning("Please select at least **3 symptoms** to enable prediction.")
        predict_disabled = True
    else:
        predict_disabled = False

    # Creating a dictionary with basic descriptions and symptoms for each disease.
    disease_info = {
        "(vertigo) Paroymsal Positional Vertigo": {
            "description": "A disorder arising in the inner ear that causes brief episodes of vertigo.",
            "symptoms": ["Dizziness", "Nausea", "Loss of balance", "Spinning sensation"],
            "management": "Avoid quick head movements and rest when dizzy."
        },
        "AIDS": {
            "description": "A chronic condition caused by HIV that damages the immune system.",
            "symptoms": ["Weight loss", "Frequent infections", "Fatigue", "Night sweats"],
            "management": "Take your meds daily and stay away from infections."
        },
        "Acne": {
            "description": "A skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
            "symptoms": ["Pimples", "Blackheads", "Whiteheads", "Skin irritation"],
            "management": "Wash gently, avoid touching your face, and use acne cream."
        },
        "Alcoholic hepatitis": {
            "description": "Liver inflammation caused by excessive alcohol intake.",
            "symptoms": ["Jaundice", "Fatigue", "Abdominal pain", "Nausea"],
            "management": "Stop drinking alcohol and eat healthy."
        },
        "Allergy": {
            "description": "A reaction by the immune system to a foreign substance.",
            "symptoms": ["Sneezing", "Itchy eyes", "Rash", "Swelling"],
            "management": "Avoid triggers and use allergy meds if needed."
        },
        "Arthritis": {
            "description": "Inflammation of one or more joints causing pain and stiffness.",
            "symptoms": ["Joint pain", "Swelling", "Reduced motion", "Stiffness"],
            "management": "Stay active, use hot/cold packs, and take pain relief if needed."
        },
        "Bronchial Asthma": {
            "description": "A condition where the airways narrow and swell and may produce extra mucus.",
            "symptoms": ["Shortness of breath", "Chest tightness", "Wheezing", "Coughing"],
            "management": "Use your inhaler and avoid smoke or dust."
        },
        "Cervical spondylosis": {
            "description": "Age-related wear and tear affecting the spinal disks in the neck.",
            "symptoms": ["Neck pain", "Stiffness", "Headaches", "Numbness"],
            "management": "Do neck exercises and avoid poor posture."
        },
        "Chicken pox": {
            "description": "A highly contagious viral infection causing an itchy, blister-like rash.",
            "symptoms": ["Fever", "Rash", "Fatigue", "Loss of appetite"],
            "management": "Rest, don’t scratch, and stay away from others."
        },
        "Chronic cholestasis": {
            "description": "A condition where bile cannot flow from the liver to the duodenum.",
            "symptoms": ["Jaundice", "Itching", "Dark urine", "Pale stool"],
            "management": "Eat low-fat foods and follow your doctor’s advice."
        },
        "Common Cold": {
            "description": "A viral infectious disease affecting the upper respiratory tract.",
            "symptoms": ["Runny nose", "Sneezing", "Sore throat", "Cough"],
            "management": "Rest, drink fluids, and use cold medicine if needed."
        },
        "Dengue": {
            "description": "A mosquito-borne viral infection causing flu-like illness.",
            "symptoms": ["High fever", "Severe headache", "Pain behind eyes", "Rash"],
            "management": "Drink lots of water and rest. Go to the doctor if it gets worse."
        },
        "Diabetes": {
            "description": "A chronic disease that affects how your body turns food into energy.",
            "symptoms": ["Increased thirst", "Frequent urination", "Fatigue", "Blurred vision"],
            "management": "Watch your sugar levels and eat healthy."
        },
        "Dimorphic hemmorhoids(piles)": {
            "description": "Swollen veins in the lowest part of your rectum and anus.",
            "symptoms": ["Pain during bowel movements", "Itching", "Swelling", "Bleeding"],
            "management": "Eat more fiber, drink water, and avoid straining."
        },
        "Drug Reaction": {
            "description": "An unintended or harmful reaction to a medication.",
            "symptoms": ["Rash", "Swelling", "Fever", "Anaphylaxis"],
            "management": "Stop the drug and see a doctor right away."
        },
        "Fungal infection": {
            "description": "Infections caused by fungi, commonly affecting skin, nails, or lungs.",
            "symptoms": ["Redness", "Itching", "Peeling skin", "Discoloration"],
            "management": "Keep the area clean and dry. Use antifungal cream."
        },
        "GERD": {
            "description": "A chronic digestive disease where stomach acid irritates the food pipe lining.",
            "symptoms": ["Heartburn", "Regurgitation", "Chest pain", "Difficulty swallowing"],
            "management": "Avoid spicy food and don’t lie down after eating."
        },
        "Gastroenteritis": {
            "description": "An intestinal infection marked by diarrhea, cramps, nausea, and fever.",
            "symptoms": ["Watery diarrhea", "Abdominal pain", "Vomiting", "Fever"],
            "management": "Drink fluids and rest. Eat light food."
        },
        "Heart attack": {
            "description": "A blockage of blood flow to the heart muscle.",
            "symptoms": ["Chest pain", "Shortness of breath", "Sweating", "Nausea"],
            "management": "Call emergency services immediately. Rest and stay calm."
        },
        "Hepatitis B": {
        "description": "A serious liver infection caused by the hepatitis B virus.",
        "symptoms": ["Jaundice", "Fatigue", "Abdominal pain", "Dark urine"],
        "management": "Avoid alcohol, eat healthy, and follow doctor’s advice."
        },
        "Hepatitis C": {
            "description": "A viral infection that causes liver inflammation and damage.",
            "symptoms": ["Fatigue", "Nausea", "Jaundice", "Muscle aches"],
            "management": "Get regular check-ups and avoid alcohol."
        },
        "Hepatitis D": {
            "description": "A liver infection caused by the hepatitis D virus, only occurs in those infected with HBV.",
            "symptoms": ["Jaundice", "Fatigue", "Abdominal pain", "Vomiting"],
            "management": "Manage hepatitis B properly and avoid alcohol."
        },
        "Hepatitis E": {
            "description": "A liver disease caused by the hepatitis E virus, usually through contaminated water.",
            "symptoms": ["Jaundice", "Loss of appetite", "Nausea", "Fever"],
            "management": "Drink clean water and get plenty of rest."
        },
        "Hypertension": {
            "description": "A condition in which the blood pressure in the arteries is persistently elevated.",
            "symptoms": ["Headaches", "Dizziness", "Blurred vision", "Nosebleeds"],
            "management": "Eat less salt, stay active, and take medicine if needed."
        },
        "Hyperthyroidism": {
            "description": "The overproduction of thyroid hormones by the thyroid gland.",
            "symptoms": ["Weight loss", "Rapid heartbeat", "Nervousness", "Sweating"],
            "management": "Take prescribed meds and avoid stress."
        },
        "Hypoglycemia": {
            "description": "A condition caused by low blood sugar levels.",
            "symptoms": ["Shakiness", "Sweating", "Confusion", "Irritability"],
            "management": "Eat or drink something sugary right away."
        },
        "Hypothyroidism": {
            "description": "A condition in which the thyroid gland doesn't produce enough hormones.",
            "symptoms": ["Fatigue", "Weight gain", "Depression", "Cold sensitivity"],
            "management": "Take your thyroid medicine daily."
        },
        "Impetigo": {
            "description": "A highly contagious skin infection causing red sores.",
            "symptoms": ["Red sores", "Itching", "Fluid-filled blisters", "Crusting skin"],
            "management": "Keep skin clean and use antibiotic cream."
        },
        "Jaundice": {
            "description": "A condition that causes yellowing of the skin and eyes due to high bilirubin.",
            "symptoms": ["Yellow skin", "Dark urine", "Fatigue", "Abdominal pain"],
            "management": "Rest, eat light food, and stay hydrated."
        },
        "Malaria": {
            "description": "A disease caused by a plasmodium parasite, transmitted by mosquitoes.",
            "symptoms": ["Fever", "Chills", "Sweating", "Headache"],
            "management": "Take antimalarial meds and rest. Avoid mosquito bites."
        },
        "Migraine": {
            "description": "A headache of varying intensity, often accompanied by nausea and sensitivity to light.",
            "symptoms": ["Throbbing pain", "Nausea", "Visual disturbances", "Sensitivity to light"],
            "management": "Rest in a quiet dark room and take pain relievers."
        },
        "Osteoarthristis": {
            "description": "A type of arthritis that occurs when flexible tissue at the ends of bones wears down.",
            "symptoms": ["Joint pain", "Stiffness", "Swelling", "Reduced flexibility"],
            "management": "Stay active, use joint support, and take pain meds if needed."
        },
        "Paralysis (brain hemorrhage)": {
            "description": "Loss of muscle function due to brain bleeding.",
            "symptoms": ["Inability to move limbs", "Speech difficulties", "Vision issues", "Confusion"],
            "management": "Call emergency help. Long-term care may include therapy."
        },
        "Peptic ulcer diseae": {
            "description": "Sores that develop on the lining of the stomach or duodenum.",
            "symptoms": ["Stomach pain", "Bloating", "Heartburn", "Nausea"],
            "management": "Avoid spicy food and take medicine as advised."
        },
        "Pneumonia": {
            "description": "Infection that inflames the air sacs in one or both lungs.",
            "symptoms": ["Cough", "Fever", "Shortness of breath", "Chest pain"],
            "management": "Take antibiotics if prescribed and get plenty of rest."
        },
        "Psoriasis": {
            "description": "A skin disease that causes red, itchy scaly patches.",
            "symptoms": ["Red patches", "Dry skin", "Itching", "Cracked skin"],
            "management": "Use moisturizing creams and avoid skin triggers."
        },
        "Tuberculosis": {
            "description": "A potentially serious infectious bacterial disease that mainly affects the lungs.",
            "symptoms": ["Persistent cough", "Weight loss", "Fever", "Night sweats"],
            "management": "Take your TB meds every day without missing a dose."
        },
        "Typhoid": {
            "description": "A bacterial infection due to Salmonella typhi.",
            "symptoms": ["High fever", "Weakness", "Stomach pain", "Rash"],
            "management": "Take antibiotics and drink clean water."
        },
        "Urinary tract infection": {
            "description": "Infection in any part of the urinary system.",
            "symptoms": ["Burning sensation", "Frequent urination", "Cloudy urine", "Pelvic pain"],
            "management": "Drink lots of water and see a doctor for antibiotics."
        },
        "Varicose veins": {
            "description": "Swollen, twisted veins that you can see just under the surface of the skin.",
            "symptoms": ["Aching legs", "Swollen ankles", "Itching", "Muscle cramping"],
            "management": "Avoid standing too long and try leg-elevation or compression socks."
        },
        "hepatitis A": {
            "description": "A highly contagious liver infection caused by the hepatitis A virus.",
            "symptoms": ["Fatigue", "Nausea", "Abdominal pain", "Loss of appetite"],
            "management": "Get plenty of rest and avoid fatty foods."
        }
    }



    # Prediction logic
    if st.button("Predict Disease", type="primary", disabled=predict_disabled):
        with st.spinner("Analyzing symptoms..."):
            try:
                # Prepare input data
                symptom_values = [1 if input_data[s] else 0 for s in symptoms]
                
                # Create Spark DataFrame
                row = Row(*symptoms)
                df = spark.createDataFrame([row(*symptom_values)])
                
                # Make prediction
                result = model.transform(df)
                probability_vector = result.select("probability").collect()[0][0]
                label_list = model.stages[0].labels

                # Get top 5 predictions
                top5 = sorted(
                    enumerate(probability_vector),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                import pandas as pd
                top5_df = pd.DataFrame([
                    {"Disease": label_list[idx], "Probability": prob}
                    for idx, prob in top5
                ])

                # Show results
                st.balloons()
                st.success("**Top 5 Predicted Diseases:**")

                for idx, prob in top5:
                    disease_name = label_list[idx]
                    probability = prob

                    # Fetch disease details
                    disease_data = disease_info.get(disease_name, {})
                    description = disease_data.get('description', 'No description available.')
                    actual_symptoms = disease_data.get('symptoms', ['No symptoms listed.'])
                    advice = disease_data.get('management','No advices available.')

                    symptoms_str = "\n".join(f"- {symptom}" for symptom in actual_symptoms)

                    # Show name with a help icon and expander
                    st.markdown(f"**{disease_name}** - {probability:.1%}",
                                    help=f"{description}  \nCommon symptoms include:  \n{symptoms_str}  \n\n:blue-background[Here are some friendly advices: {advice}]"
                                )

                
            except Exception as e:
                st.error(f"""
                Prediction failed: {str(e)}  
                Please check:  
                - Model is properly loaded  
                - All symptom columns match training data
                """)

    # Footer
    st.markdown("---")
    st.markdown("AI Models isn not 100% accurate. Consult your Doctor")
with tabs[1]:
    st.image('/opt/spark/output/class_distribution.png', caption='Class Distribution', width=1200)
    st.image('/opt/spark/output/confusion_matrix.png', caption='Confusion Matrix', width=1200)
    st.image('/opt/spark/output/evaluation_metrics.png', caption='Evaluation Metrics', width=1200)
