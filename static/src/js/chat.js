const aboutme_div = document.querySelector(".aboutme-div");
const background_div = document.querySelector(".homepage-background");
const circle_divs = document.querySelectorAll(".home-circle-div");

function sendMessage() {
    var userInput = document.getElementById('user-input');
    var message = userInput.value.trim();
    var systemPrompt = document.getElementById('system-prompt').value;
    
    if (message === '') return;
    
    addMessage(message, 'user-message');
    userInput.value = '';
    
    fetch('/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            user_input: message,
            system_prompt: systemPrompt
        }),
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response, 'bot-message');
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Did you fill in your API key?', 'bot-message error');
    });
}

function addMessage(text, className) {
    var chatMessages = document.getElementById('chat-messages');
    var messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + className;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Allow sending message with Enter key
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Ensure the send button also triggers sendMessage
document.getElementById('send-button').addEventListener('click', sendMessage);

function adaptDivHeight(){
    var div_height = aboutme_div.offsetHeight;
    if (window.innerWidth > 1430){ // If the image is next to the text
        if (div_height > 0.82*window.innerHeight){
            var new_background_height = div_height/0.82;
            background_div.style.height = new_background_height + "px"; 
        }
        else{
            background_div.style.height = "100%";
        }
    }
    else { // If the image is under the text
        background_div.style.removeProperty("height");
    }
    circle_divs.forEach(circle_div => {
        circle_div.style.height = background_div.offsetHeight + "px";
    });
}

function onLoad(){
    adaptDivHeight();
}

window.addEventListener("DOMContentLoaded", onLoad);
window.addEventListener("resize", adaptDivHeight);