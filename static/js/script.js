// HTML elements selection
const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");
const uploadArea = document.getElementById("uploadArea");
const uploadProgress = document.getElementById("uploadProgress");
const uploadError = document.getElementById("uploadError");
const processingSection = document.getElementById("processingSection");
const resultsSection = document.getElementById("resultsSection");
const personalInfoTable = document.getElementById("personalInfoTable");
const summaryText = document.getElementById("summaryText");
const technicalSkillsList = document.getElementById("technicalSkillsList");
const softSkillsList = document.getElementById("softSkillsList");
const experienceList = document.getElementById("experienceList");
const educationList = document.getElementById("educationList");
const certificationsList = document.getElementById("certificationsList");
const projectsList = document.getElementById("projectsList");
const languagesList = document.getElementById("languagesList");
const showJsonBtn = document.getElementById("showJsonBtn");
const jsonModal = new bootstrap.Modal(document.getElementById("jsonModal"));
const jsonContent = document.getElementById("jsonContent");
const copyJsonBtn = document.getElementById("copyJsonBtn");

// Function to activate upload
browseButton.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
        startUpload(file);
    } else {
        showError("Please upload a PDF file.");
    }
});

// Function for uploading and processing the resume
async function startUpload(file) {
    // Reset errors and progress
    uploadError.classList.add("d-none");
    uploadProgress.classList.remove("d-none");
    uploadProgress.querySelector(".progress-bar").style.width = "0%";

    const formData = new FormData();
    formData.append("file", file);

    // Display processing section
    processingSection.classList.remove("d-none");

    try {
        // Send file to backend
        const response = await fetch("/api/process/cv", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Error uploading the file.");
        }

        // Update progress bar
        uploadProgress.querySelector(".progress-bar").style.width = "100%";

        // Get JSON response
        const result = await response.json();
        displayResults(result);

    } catch (error) {
        showError(error.message);
    } finally {
        // Hide processing indicator
        processingSection.classList.add("d-none");
    }
}

// Function to display results
function displayResults(data) {
    resultsSection.classList.remove("d-none");

    // Personal information
    personalInfoTable.innerHTML = generateTableRows(data.personalInfo);

    // Summary
    summaryText.textContent = data.summary || "No summary found.";

    // Skills
    generateSkills(data.skills);

    // Professional experiences
    generateExperience(data.experiences);

    // Education
    generateEducation(data.education);

    // Other information
    generateOtherInfo(data.other);
}

// Function to generate table rows
function generateTableRows(data) {
    return data.map(item => {
        return `<tr>
                    <td>${item.label}</td>
                    <td>${item.value}</td>
                </tr>`;
    }).join("");
}

// Function to generate skills
function generateSkills(skills) {
    technicalSkillsList.innerHTML = generateSkillList(skills.technical);
    softSkillsList.innerHTML = generateSkillList(skills.soft);
}

// Function to generate a list of skills
function generateSkillList(skills) {
    return skills.map(skill => `<li>${skill}</li>`).join("");
}

// Function to generate professional experiences
function generateExperience(experiences) {
    experienceList.innerHTML = experiences.map(exp => {
        return `<div>
                    <strong>${exp.jobTitle}</strong> at ${exp.companyName}<br>
                    <em>${exp.dateRange}</em>
                    <p>${exp.description}</p>
                </div>`;
    }).join("");
}

// Function to generate education
function generateEducation(education) {
    educationList.innerHTML = education.map(edu => {
        return `<div>
                    <strong>${edu.degree}</strong> in ${edu.fieldOfStudy} at ${edu.institution}<br>
                    <em>${edu.dateRange}</em>
                </div>`;
    }).join("");
}

// Function to generate other information
function generateOtherInfo(other) {
    // Certifications
    certificationsList.innerHTML = other.certifications.map(cert => `<div>${cert}</div>`).join("");

    // Projects
    projectsList.innerHTML = other.projects.map(project => `<div>${project}</div>`).join("");

    // Languages
    languagesList.innerHTML = other.languages.map(language => `<li>${language}</li>`).join("");
}

// Display an error
function showError(message) {
    uploadError.classList.remove("d-none");
    uploadError.textContent = message;
    uploadProgress.classList.add("d-none");
}

// Display raw JSON in a modal
showJsonBtn.addEventListener("click", () => {
    jsonContent.textContent = JSON.stringify(resultsSection, null, 2);
    jsonModal.show();
});

// Copy raw JSON
copyJsonBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(jsonContent.textContent).then(() => {
        alert("JSON copied to clipboard.");
    });
});