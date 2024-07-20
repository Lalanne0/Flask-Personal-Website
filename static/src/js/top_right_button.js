const navbar = document.querySelector(".mobile-navbar");
var isDisplayed = false;

function handleNavbarDisplay(){
    if (isDisplayed){ // If it on screen, it is closed
        navbar.style.transform = "translateX(100%)";
    }
    else { // If it is not on screen, it is opened
        navbar.style.transform = "translateX(0)";
    }
    isDisplayed = !isDisplayed;
}

function displayLanguageError() {
    const errorMessage = document.getElementById('language-error-message');
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 3000); // Hide the message after 3 seconds
}

function handleTopRightClick() {
    if (window.innerWidth > 800) {
        // Desktop version
        displayLanguageError();
    } else {
        // Mobile version
        const mobileNavbar = document.querySelector('.mobile-navbar');
        if (mobileNavbar.style.transform === 'translateX(0px)') {
            mobileNavbar.style.transform = 'translateX(100%)';
        } else {
            mobileNavbar.style.transform = 'translateX(0)';
        }
    }
}

function handleMobileLanguageClick() {
    displayLanguageError();
    // Close the mobile navbar after displaying the error
    document.querySelector('.mobile-navbar').style.transform = 'translateX(100%)';
}