const aboutme_div = document.querySelector(".aboutme-div");
const background_div = document.querySelector(".homepage-background");
const circle_divs = document.querySelectorAll(".home-circle-div");

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