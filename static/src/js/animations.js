const scrollOffsetPercentage = 0.15;
const animatedHomepage = document.querySelector(".js-animation-homepage");
const animatedElements1 = Array.from(document.getElementsByClassName("js-animation-1"));
const animatedElements2 = Array.from(document.getElementsByClassName("js-animation-2"));
const animatedContact = document.querySelector(".js-animation-contact");
const animatedAboutMe = document.querySelector(".js-animation-aboutme");
const animatedProjectsTitle = document.querySelector(".js-animation-projects-title");
const animatedArticleTiles = document.querySelectorAll(".article-tile");

function elementInViewport(el) {
  const height = el.getBoundingClientRect().top;
  return (window.innerHeight - height) / window.innerHeight  > scrollOffsetPercentage;
}

function displayHomepage() {
    setTimeout( () => animatedHomepage.children[0].children[0].classList.add('js-visible'), 0);
    setTimeout( () => animatedHomepage.children[0].children[1].classList.add('js-visible'), 200);
    setTimeout( () => animatedHomepage.children[1].classList.add('js-visible'), 400);
}

function displayElement1(el) {
    setTimeout( () => el.children[0].classList.add('js-visible'), 0);
    setTimeout( () => el.children[1].classList.add('js-visible'), 200);
}

function displayContact() {
    setTimeout( () => animatedContact.children[0].children[0].classList.add('js-visible'), 0);
    setTimeout( () => animatedContact.children[0].children[1].classList.add('js-visible'), 100);
    setTimeout( () => animatedContact.children[1].classList.add('js-visible'), 400);
    setTimeout( () => animatedContact.children[2].classList.add('js-visible'), 500);
    setTimeout( () => animatedContact.children[3].classList.add('js-visible'), 600);
}


function displayAboutMe(){
    setTimeout( () => animatedAboutMe.children[0].children[0].classList.add('js-visible'), 0); // About me title
    setTimeout( () => animatedAboutMe.children[1].classList.add('js-visible'), 0); // Photo
    setTimeout( () => animatedAboutMe.children[0].children[1].children[0].classList.add('js-visible'), 100); // Description
    setTimeout( () => animatedAboutMe.children[0].children[1].children[1].classList.add('js-visible'), 200); // Description
    setTimeout( () => animatedAboutMe.children[0].children[1].children[2].classList.add('js-visible'), 300); // Description
    setTimeout( () => animatedAboutMe.children[0].children[1].children[3].classList.add('js-visible'), 400); // Description
    setTimeout( () => animatedAboutMe.children[0].children[1].children[4].classList.add('js-visible'), 500); // Description
}

function displayProjectsTitle(){
    setTimeout( () => animatedProjectsTitle.children[0].classList.add('js-visible'), 0);
    setTimeout( () => animatedProjectsTitle.children[1].classList.add('js-visible'), 200);
}

function HandleAnimateIndex(){

    animatedElements1.forEach(el => {
        if (elementInViewport(el)) {
            displayElement1(el);
        }
    });
    
    var j;
    for (var i = 0; i < animatedElements2.length; i++) {
        if (window.innerWidth > 1420){
            if ((i%3 == 0) && elementInViewport(animatedElements2[i])) {
                animatedElements2[i].classList.add('js-visible');
                j = i+1;
                setTimeout( () => animatedElements2[j].classList.add('js-visible'), 200);
                setTimeout( () => animatedElements2[j+1].classList.add('js-visible'), 400);
            }
        }
        else if(window.innerWidth > 980){
            if ((i%2 == 0) && elementInViewport(animatedElements2[i])) {
                animatedElements2[i].classList.add('js-visible');
                j = i+1;
                setTimeout( () => animatedElements2[j].classList.add('js-visible'), 200);
            }
        }
        else {
            if (elementInViewport(animatedElements2[i])) {
                animatedElements2[i].classList.add('js-visible');
            }
        }
    }

    if (elementInViewport(animatedContact)) {
        displayContact();
    }
}

function HandleAnimateAboutMe(){
    for (var i = 0; i < animatedElements2.length; i++) {
        if (window.innerWidth > 1542){ // If the 3 elements are next to each other
            if (elementInViewport(animatedElements2[i])) {
                animatedElements2[0].classList.add('js-visible');
                setTimeout( () => animatedElements2[1].classList.add('js-visible'), 200);
                setTimeout( () => animatedElements2[2].classList.add('js-visible'), 400);
            }
        }
        else if(window.innerWidth > 1028){ // If 2 elements are next to each other and the 3rd is alone
            if ((i == 0) && elementInViewport(animatedElements2[i])) { // If the 1st is in viewport
                animatedElements2[i].classList.add('js-visible'); // Display the first 2
                setTimeout( () => animatedElements2[1].classList.add('js-visible'), 200);
            }
            else if ((i == 2) && elementInViewport(animatedElements2[i])) { // If the 3rd is in viewport
                animatedElements2[i].classList.add('js-visible'); // Display it alone
            }
        }
        else { // If elements are in column
            if (elementInViewport(animatedElements2[i])) {
                animatedElements2[i].classList.add('js-visible');
            }
        }
    }
}

function HandleAnimateArticleTiles(){
    animatedArticleTiles.forEach(el => {
        if (elementInViewport(el)) {
            el.classList.add('js-visible');
        }
    });
}

if (currentPage === "index") {
    window.addEventListener('DOMContentLoaded', () => {
        displayHomepage();
        setTimeout( () => HandleAnimateIndex(), 500);
    });

    window.addEventListener('scroll', () => {
        HandleAnimateIndex();
    });
}
else if (currentPage === "aboutme") {
    window.addEventListener('DOMContentLoaded', () => {
        displayAboutMe();
    });

    window.addEventListener('scroll', () => {
        HandleAnimateAboutMe();
    });
}

else if (currentPage == "projects") {
    window.addEventListener('DOMContentLoaded', () => {
        displayProjectsTitle();
        setTimeout( () => HandleAnimateArticleTiles(), 200); // The first elements are visible directly
    });

    window.addEventListener('scroll', () => {
        HandleAnimateArticleTiles();
    });
}