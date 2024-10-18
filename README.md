# Flask-Personal-Website

## Calling and using Claude's API with custom parameters

This project presents a comprehensive example of a Flask environment that uses some AI tools. It calls Anthropic's API for Claude, and creates several models on the Titanic dataset. Detailed results of possible experimentations with Claude's API are presented in the `report.pdf` file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lalanne0/Flask-Personal-Website.git
```

2. Set the terminal in the repository:
```bash
cd Flask-Personal-Website
```

3. Create a virtual environment:
```bash
python -m venv env
```

4. Activate the environment:
```bash
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

6. If you want to use Claude's API, set up your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'  # On Windows use `setx ANTHROPIC_API_KEY "your-api-key-here"`
```

7. Run the Flask application:
```bash
python app.py
```

This may take a few moments. The app will train several AI models before starting.

8. Visit `http://localhost:5000` in your web browser.

9. When you finished using the platform, deactivate the environment
```bash
deactivate
```

## Acknowledments

This project uses the Titanic dataset available on [Kaggle](https://www.kaggle.com/competitions/titanic/data).