from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import anthropic

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

@app.route('/articles/<path:filename>')
def serve_article(filename):
    return send_from_directory('static/articles', filename)

@app.route('/cv/<path:filename>')
def serve_cv(filename):
    return send_from_directory('static/cv', filename)

if __name__ == '__main__':
    app.run(debug=True)