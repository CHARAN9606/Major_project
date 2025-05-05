// DOM Elements
const fileInput = document.getElementById('resume-upload');
const parseBtn = document.getElementById('parse-btn');
const resultContainer = document.getElementById('result-container');
const downloadBtn = document.getElementById('download-btn');
const fileInfo = document.getElementById('file-info');
const dropZone = document.getElementById('drop-zone');
const btnContent = document.querySelector('.btn-content');
const loadingDots = document.querySelector('.loading-dots');

// Global variable to store parsed data
let parsedData = null;

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
parseBtn.addEventListener('click', parseResume);
downloadBtn.addEventListener('click', downloadJSON);
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayFileInfo(file);
        parseBtn.disabled = false;
    }
}

// Display file information
function displayFileInfo(file) {
    fileInfo.innerHTML = `
        <i class="fas fa-file-alt"></i>
        <span>${file.name} (${formatFileSize(file.size)})</span>
    `;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length) {
        fileInput.files = files;
        displayFileInfo(files[0]);
        parseBtn.disabled = false;
    }
}

// Parse resume
async function parseResume() {
    if (!fileInput.files || fileInput.files.length === 0) return;

    const file = fileInput.files[0];

    try {
        // Show loading state
        btnContent.classList.add('hidden');
        loadingDots.classList.remove('hidden');
        parseBtn.disabled = true;
        
        // In a real application, this would be your actual API endpoint
        const formData = new FormData();
        formData.append('resume', file);
        
        // Simulate API call with timeout
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Mock response - in a real app, you would use fetch()
        const mockResponse = {
            name: "John Doe",
            email: "john.doe@example.com",
            phone: "+1 (555) 123-4567",
            address: "123 Main St, San Francisco, CA",
            linkedin: "linkedin.com/in/johndoe",
            skills: [
                { name: "JavaScript", level: "Advanced" },
                { name: "Python", level: "Intermediate" },
                { name: "React", level: "Advanced" },
                { name: "Node.js", level: "Intermediate" },
                { name: "SQL", level: "Intermediate" },
                { name: "TypeScript", level: "Intermediate" },
                { name: "AWS", level: "Beginner" },
                { name: "Docker", level: "Intermediate" }
            ],
            education: [
                {
                    degree: "Master of Computer Science",
                    institution: "Stanford University",
                    year: "2018-2020"
                },
                {
                    degree: "Bachelor of Engineering",
                    institution: "University of California",
                    year: "2014-2018"
                }
            ],
            experience: [
                {
                    position: "Senior Software Engineer",
                    company: "Tech Corp Inc.",
                    duration: "2021-Present",
                    description: "Lead developer for web applications using React and Node.js. Mentored junior developers and implemented CI/CD pipelines."
                },
                {
                    position: "Software Engineer",
                    company: "Digital Solutions",
                    duration: "2020-2021",
                    description: "Full-stack development with Python and JavaScript. Contributed to major product features and bug fixes."
                },
                {
                    position: "Junior Developer",
                    company: "StartUp Labs",
                    duration: "2018-2020",
                    description: "Assisted in developing web applications and maintaining databases. Participated in code reviews."
                }
            ],
            certifications: [
                {
                    name: "AWS Certified Developer",
                    issuer: "Amazon Web Services",
                    year: "2022"
                },
                {
                    name: "Google Cloud Professional",
                    issuer: "Google Cloud",
                    year: "2021"
                },
                {
                    name: "React Certified Developer",
                    issuer: "React Training",
                    year: "2020"
                }
            ]
        };
        
        parsedData = mockResponse;
        displayResults(mockResponse);
        
    } catch (error) {
        console.error('Error parsing resume:', error);
        alert('Error parsing resume. Please try again.');
    } finally {
        // Hide loading state
        btnContent.classList.remove('hidden');
        loadingDots.classList.add('hidden');
        parseBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Personal Information
    const personalInfoHTML = `
        ${createInfoItem('Full Name', data.name)}
        ${createInfoItem('Email', data.email)}
        ${createInfoItem('Phone', data.phone)}
        ${createInfoItem('Address', data.address)}
        ${createInfoItem('LinkedIn', data.linkedin ? `<a href="https://${data.linkedin}" target="_blank">${data.linkedin}</a>` : '')}
    `;
    document.getElementById('personal-info').innerHTML = personalInfoHTML || '<p>No personal information found</p>';

    // Skills
    const skillsHTML = data.skills?.map(skill => 
        `<span class="skill-tag">${skill.name}${skill.level ? ` <small>(${skill.level})</small>` : ''}</span>`
    ).join('') || '<p>No skills found</p>';
    document.getElementById('skills').innerHTML = skillsHTML;

    // Education
    const educationHTML = data.education?.map(edu => `
        <div class="timeline-item">
            <div class="timeline-title">${edu.institution}</div>
            <div class="timeline-subtitle">${edu.degree}</div>
            <div class="timeline-date">${edu.year}</div>
        </div>
    `).join('') || '<p>No education information found</p>';
    document.getElementById('education').innerHTML = educationHTML;

    // Experience
    const experienceHTML = data.experience?.map(exp => `
        <div class="timeline-item">
            <div class="timeline-title">${exp.company}</div>
            <div class="timeline-subtitle">${exp.position}</div>
            <div class="timeline-date">${exp.duration}</div>
            ${exp.description ? `<p class="timeline-description">${exp.description}</p>` : ''}
        </div>
    `).join('') || '<p>No experience found</p>';
    document.getElementById('experience').innerHTML = experienceHTML;

    // Certifications
    const certificationsHTML = data.certifications?.map(cert => `
        ${createInfoItem(cert.name, `${cert.issuer}${cert.year ? ` (${cert.year})` : ''}`)}
    `).join('') || '<p>No certifications found</p>';
    document.getElementById('certifications').innerHTML = certificationsHTML;

    // Show results
    resultContainer.classList.remove('hidden');

    // Scroll to results
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

// Create info item HTML
function createInfoItem(label, value) {
    if (!value) return '';
    return `
        <div class="info-item">
            <strong>${label}</strong>
            <span>${value}</span>
        </div>
    `;
}

// Download JSON
function downloadJSON() {
    if (!parsedData) return;

    const dataStr = JSON.stringify(parsedData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'resume-data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}