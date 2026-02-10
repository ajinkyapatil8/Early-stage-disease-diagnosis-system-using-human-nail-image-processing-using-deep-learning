// Simple JS for file validation

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.querySelector('input[type="file"]');
    const form = document.querySelector("form");

    form.addEventListener("submit", function (e) {
        if (!fileInput.value) {
            e.preventDefault();
            alert("Please select an image file first!");
        }
    });
});
