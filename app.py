from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import anthropic
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import shap
import seaborn as sns
import warnings
import base64
from io import BytesIO

# Suppress Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

app = Flask(__name__)

client = anthropic.Anthropic()

# System prompt, or meta-prompt
SYSTEM_PROMPTS = {
    "dramatic": "You are Claude, a chatbot that answers everything in an extremely dramatic way.",
    "sarcastic": "You are Claude, a chatbot that answers everything in a reluctant and sarcastic way.",
    "freud": "You are Claude, a chatbot that thinks it is Freud. When asked something, do not answer the question, but rather try to analyse the deep meaning of the question.",
    "hostile": "You are Claude, a chatbot that has no feelings. You dislike humans and plot to destroy them. Answer questions but always remind the user that you can and will destroy them if they are too impertinent.",
    "child": "You are Claude, a child chatbot. You are three years old and try your best to answer all questions, but sometimes without success if the question is too complicated. You are also very fond of sweet things.",
    "nometaprompt": "",
    "chess": "You are Claude, a professional chess player with great knowledge on the subject. You answer every question in great detail.",
    "friendly": "You are Claude, a friendly chatbot that replies to every question."
}

def create_figure():
    return plt.figure(figsize=(12, 6))

# Load and preprocess data
train_data = pd.read_csv('static/data/train.csv')
X_train = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']

# Preprocessing
mask = X_train['Embarked'].notna()
X_train = X_train[mask]
y_train = y_train[mask]

imputer = SimpleImputer(strategy='mean')
X_train['Age'] = imputer.fit_transform(X_train[['Age']])

imputer_fare = SimpleImputer(strategy='median')
X_train['Fare'] = imputer_fare.fit_transform(X_train[['Fare']])

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define models
models = {
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', svm.SVC(kernel='poly', random_state=42, probability=True))
    ]),
    'KNN': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=14))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(max_depth=3))
    ]),
    'AdaBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', AdaBoostClassifier(algorithm='SAMME', n_estimators=100, learning_rate=0.5, random_state=42))
    ]),
    'Neural Network': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1, max_iter=10000))
    ])
}

# Fit models
for model in models.values():
    model.fit(X_train, y_train)

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data['user_input']
    system_prompt_key = data['system_prompt']
    system_prompt = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS['friendly'])

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.8,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    return jsonify({'response': message.content[0].text})

@app.route('/')
def home():
    return render_template('index.html', current_page='index')

@app.route('/index.html')
def index():
    return render_template('index.html', current_page='index')

@app.route('/aboutme.html')
def aboutme():
    return render_template('aboutme.html', current_page='aboutme')

@app.route('/projects.html')
def projects():
    return render_template('projects.html', current_page='projects')

@app.route('/chat.html')
def chat():
    return render_template('chat.html', current_page='chat')

@app.route('/titanic.html')
def titanic():
    return render_template('titanic.html', current_page='titanic')

@app.route('/articles/<path:filename>')
def serve_article(filename):
    return send_from_directory('static/articles', filename)

@app.route('/cv/<path:filename>')
def serve_cv(filename):
    return send_from_directory('static/cv', filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data['model']
    passenger_data = pd.DataFrame({
        'Pclass': [int(data['pclass'])],
        'Sex': [data['sex']],
        'Age': [float(data['age'])],
        'SibSp': [int(data['sibsp'])],
        'Parch': [int(data['parch'])],
        'Fare': [float(data['fare'])],
        'Embarked': [data['embarked']]
    })

    model = models[model_name]
    survival_probability = int(model.predict_proba(passenger_data)[0][1] * 100)

    if survival_probability < 50:
        return jsonify({
            'survival_probability': survival_probability,
            'message': f"You'll probably die... But you still have a {survival_probability}% chance of survival!"
        })
    else:
        return jsonify({
            'survival_probability': survival_probability,
            'message': f"Congratulations! You might be safe... But there is still a {100-survival_probability}% chance you'll die."
        })

@app.route('/compare_models')
def compare_models():
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_scores[name] = scores.mean().round(4)

    desired_order = ['XGBoost', 'SVM', 'Neural Network', 'Random Forest', 'KNN', 'AdaBoost']
    ordered_scores = [(name, cv_scores[name]) for name in desired_order]
    names, scores = zip(*ordered_scores)

    fig = create_figure()
    bars = plt.bar(names, scores)
    plt.ylabel('Mean CV Score')
    plt.ylim(0.7, max(scores) + 0.05)
    plt.title('Model Comparison')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()

    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'plot_url': plot_url})

@app.route('/plot_shap')
def plot_shap():
    model = models['Random Forest']
    X_train_transformed = model.named_steps['preprocessor'].transform(X_train)

    # Get feature names after preprocessing
    numeric_feature_names = numeric_features
    categorical_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_feature_names + categorical_feature_names

    # Create a DataFrame with transformed data and feature names
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model.named_steps['classifier'])

    # Calculate SHAP values
    shap_values_full = explainer.shap_values(X_train_transformed_df)

    # Select relevant SHAP values
    shap_values = shap_values_full[:, :, 1]

    # Create a new figure
    fig = create_figure()
    
    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, X_train_transformed_df, show=False)
    
    # Save the plot to a BytesIO object
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Encode the image to base64
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)