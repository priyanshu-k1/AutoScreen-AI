
// File upload functionality
document.addEventListener('DOMContentLoaded', function () {
    const dropArea = document.querySelector('.drop-area');
    const fileInput = document.getElementById('fileInput');
    const fileListDiv = document.getElementById('fileList');
    const browseButton = document.querySelector('.browse-button');
    const totalSizeElement = document.getElementById('totalSize');
    const uploadBtn = document.getElementById('uploadBtn');
    const progressContainer = document.getElementById('progressContainer-upload'); // Updated ID
    const progressBar = document.getElementById('progressBar-upload'); // Updated ID
    const percentComplete = document.getElementById('percentComplete');
    const uploadingFile = document.getElementById('uploadingFile');

    // Set accepted file types
    fileInput.setAttribute('accept', '.pdf,.doc,.docx');

    let files = [];
    let isUploading = false;
    const MAX_FILE_SIZE = 10 * 1024 * 1024; 
    const MAX_TOTAL_SIZE = 20 * 1024 * 1024;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    }

    // File input handlers
    browseButton.addEventListener('click', () => fileInput.click());
    dropArea.addEventListener('click', (e) => {
        if (e.target !== browseButton) fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files);
            fileInput.value = '';
        }
    });

    // File handling
    function handleFiles(newFiles) {
        if (newFiles.length === 0) return;

        const validFiles = Array.from(newFiles).filter(file => {
            const fileName = file.name.toLowerCase();
            const validType = fileName.endsWith('.pdf') || 
                            fileName.endsWith('.doc') || 
                            fileName.endsWith('.docx');
            
            if (!validType) return false;
            
            if (file.size > MAX_FILE_SIZE) {
                showValidationPopup(`Total size exceeds limit (max ${formatFileSize(MAX_TOTAL_SIZE)})`);
                return false;
            }
            
            return true;
        });

        if (validFiles.length === 0) {
            showValidationPopup('Please select only PDF or Word documents (max 10MB each)');
            return;
        }
        
        const newTotalSize = files.reduce((acc, f) => acc + f.size, 0) + 
                           validFiles.reduce((acc, f) => acc + f.size, 0);
        
        if (newTotalSize > MAX_TOTAL_SIZE) {
            showValidationPopup(`Total size exceeds limit (max ${formatFileSize(MAX_TOTAL_SIZE)})`);
            return;
        }
        
        if (validFiles.length < newFiles.length) {
            showValidationPopup('Some files were not added (invalid type or too large)');
        }
        files = [...files, ...validFiles];
        updateFileList();
        updateTotalSize();
        uploadBtn.disabled = files.length === 0 || isUploading;
    }

    function updateFileList() {
        fileListDiv.innerHTML = files.length === 0 
            ? '<div class="empty-message">No files selected</div>'
            : files.map((file, index) => `
                <div class="file-item">
                    <div class="file-info">
                        <span class="file-icon">${file.name.toLowerCase().endsWith('.pdf') ? '📕' : '📘'}</span>
                        <span class="file-name">${file.name}</span>
                        <span class="file-size">${formatFileSize(file.size)}</span>
                    </div>
                    <div class="file-actions">
                        <button class="remove-btn" data-index="${index}">✖</button>
                    </div>
                </div>
            `).join('');

        document.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                removeFile(parseInt(e.target.getAttribute('data-index')));
            });
        });
    }

    function removeFile(index) {
        files.splice(index, 1);
        updateFileList();
        updateTotalSize();
        uploadBtn.disabled = files.length === 0 || isUploading;
    }

    function updateTotalSize() {
        if (totalSizeElement) {
            const totalSize = files.reduce((acc, file) => acc + file.size, 0);
            totalSizeElement.textContent = `Total: ${formatFileSize(totalSize)}`;
        }
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Upload functionality
    uploadBtn.addEventListener('click', async function() {
        if (files.length === 0 || isUploading) return;

        isUploading = true;
        uploadBtn.disabled = true;
        
        // Reset UI
        if (progressBar) progressBar.style.width = '0%';
        if (percentComplete) percentComplete.textContent = '0%';
        if (progressContainer) progressContainer.style.display = 'block';
        if (uploadingFile) uploadingFile.textContent = `Uploading ${files.length} file(s)...`;

        try {
            // Create FormData with all files
            const formData = new FormData();
            files.forEach(file => {
                formData.append('file', file);
            });

            // Send the files to the server
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Upload response:', data);

            // Complete upload UI
            if (progressBar) progressBar.style.width = '100%';
            if (percentComplete) percentComplete.textContent = '100%';
            if (uploadingFile) uploadingFile.textContent = 'Upload complete!';

            // Check for successful processing
            if (data.success) {
                // Show success message before redirecting
                if (uploadingFile) uploadingFile.textContent = 'Upload complete! Redirecting to results...';
                
                setTimeout(() => {
                    window.location.href = '/results';
                }, 1500);
            } else {
                // If no success, show error message
                if (uploadingFile) uploadingFile.textContent = `Upload completed with ${data.errors} errors`;
                
                // Reset after delay
                setTimeout(() => {
                    files = [];
                    updateFileList();
                    updateTotalSize();
                    if (progressContainer) progressContainer.style.display = 'none';
                    isUploading = false;
                    uploadBtn.disabled = true;
                }, 2000);
            }

        } catch (error) {
            console.error('Upload failed:', error);
            if (uploadingFile) uploadingFile.textContent = `Upload failed: ${error.message}`;
            
            setTimeout(() => {
                if (progressContainer) progressContainer.style.display = 'none';
                isUploading = false;
                uploadBtn.disabled = files.length === 0;
            }, 2000);
        }
    });

    // Initialize
    updateFileList();
    updateTotalSize();
    uploadBtn.disabled = files.length === 0;
});


function showValidationPopup(message) {
    // Create popup container if it doesn't exist
    let popup = document.getElementById('validation-popup');
    
    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'validation-popup';
        popup.className = 'validation-popup';
        document.body.appendChild(popup);
    }
    
    // Set popup content
    popup.innerHTML = `
        <div class="popup-content">
            <p>${message}</p>
            <button class="popup-btn" onclick="closeValidationPopup()">OK</button>
        </div>
    `;
    
    // Show popup with animation
    setTimeout(() => {
        popup.classList.add('show');
    }, 10);
}

function closeValidationPopup() {
    const popup = document.getElementById('validation-popup');
    if (popup) {
        popup.classList.remove('show');
        setTimeout(() => {
            popup.remove();
        }, 300);
    }
}

// dashboard code 
// Dashboard functionality - Job screening features
function addCustomRole() {
    const customRoleInput = document.getElementById('custom-role');
    const roleValue = customRoleInput.value.trim();

    if (roleValue) {
        const jobRolesContainer = document.getElementById('job-roles');
        const slug = roleValue.toLowerCase().replace(/\s+/g, '-');

        const newRole = document.createElement('label');
        newRole.className = 'checkbox-item';
        newRole.innerHTML = `
            <input type="checkbox" name="role" value="${slug}" checked> ${roleValue}
        `;

        jobRolesContainer.appendChild(newRole);
        customRoleInput.value = '';

        // Add event listener to the new checkbox item
        attachCheckboxListeners(newRole);
    }
}

function addSkill() {
    const skillInput = document.getElementById('skill-input');
    const skillValue = skillInput.value.trim();

    if (skillValue) {
        const skillsContainer = document.getElementById('skills-container');

        const skillTag = document.createElement('div');
        skillTag.className = 'skill-tag';
        skillTag.innerHTML = `
            <span>${skillValue}</span>
            <button class="remove-skill" onclick="removeSkill(this)">×</button>
        `;

        skillsContainer.appendChild(skillTag);
        skillInput.value = '';
    }
}
// Handle fresher checkbox to disable years input when checked
document.addEventListener('DOMContentLoaded', function () {
    const fresherCheckbox = document.getElementById('fresher-checkbox');
    const yearsInput = document.getElementById('years-experience');

    if (fresherCheckbox && yearsInput) {
        fresherCheckbox.addEventListener('change', function () {
            if (this.checked) {
                yearsInput.value = '';
                yearsInput.disabled = true;
            } else {
                yearsInput.disabled = false;
            }
        });
    }
});
function removeSkill(button) {
    const skillTag = button.parentNode;
    skillTag.remove();
}

function attachCheckboxListeners(item) {
    item.addEventListener('click', function () {
        const checkbox = this.querySelector('input[type="checkbox"]');
        if (checkbox.checked) {
            this.classList.add('active');
        } else {
            this.classList.remove('active');
        }
    });

    const checkbox = item.querySelector('input[type="checkbox"]');
    checkbox.addEventListener('change', function () {
        if (this.checked) {
            item.parentElement.classList.add('active');
        } else {
            item.parentElement.classList.remove('active');
        }
    });
}

function redirectToUpload() {
    // Collect all selected job roles
    const jobRoles = [];
    document.querySelectorAll('#job-roles input[type="checkbox"]:checked').forEach(checkbox => {
        jobRoles.push({
            value: checkbox.value,
            label: checkbox.parentElement.textContent.trim()
        });
    });

    // Collect all added skills
    const skills = [];
    document.querySelectorAll('.skill-tag span').forEach(skillSpan => {
        skills.push(skillSpan.textContent.trim());
    });

    // Get experience information
    const isFresher = document.getElementById('fresher-checkbox').checked;
    const yearsExperience = document.getElementById('years-experience').value || 0;

    // Validate fields
    let errorMessage = "";

    if (jobRoles.length === 0) {
        errorMessage = "Please select at least one job role.";
    } else if (skills.length === 0) {
        errorMessage = "Please add at least one required skill.";
    } else if (!isFresher && (!yearsExperience || yearsExperience <= 0)) {
        errorMessage = "Please specify years of experience or select 'Fresher'.";
    }

    // If validation fails, show popup
    if (errorMessage) {
        showValidationPopup(errorMessage);
        return;
    }

    // Create the data object
    const dashboardData = {
        jobRoles: jobRoles,
        skills: skills,
        isFresher: isFresher,
        yearsExperience: yearsExperience,
        timestamp: new Date().toISOString()
    };

    // Send the data to the backend
    fetch('/save-preferences', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(dashboardData)
    })
        .then(response => response.json())
        .then(data => {
            console.log('Preferences saved:', data);  
            window.location.href = "index.html"; 
        })
        .catch((error) => {
            console.error('Error saving preferences:', error);
            showValidationPopup("There was an error saving your preferences. Please try again.");
        });
}

// Function to show validation popup
function showValidationPopup(message) {
    // Create popup container if it doesn't exist
    let popup = document.getElementById('validation-popup');

    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'validation-popup';
        popup.className = 'validation-popup';
        document.body.appendChild(popup);
    }

    // Set popup content
    popup.innerHTML = `
        <div class="popup-content">
            <p>${message}</p>
            <button class="popup-btn" onclick="closeValidationPopup()">OK</button>
        </div>
    `;

    // Show popup with animation
    setTimeout(() => {
        popup.classList.add('show');
    }, 10);
}

// Function to close validation popup
function closeValidationPopup() {
    const popup = document.getElementById('validation-popup');
    if (popup) {
        popup.classList.remove('show');
        setTimeout(() => {
            popup.remove();
        }, 300);
    }
}

// Initialize dashboard elements when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Initialize checkbox items
    const checkboxItems = document.querySelectorAll('.checkbox-item');
    if (checkboxItems.length > 0) {
        checkboxItems.forEach(item => {
            attachCheckboxListeners(item);
        });
    }

    // Add event listeners to dashboard buttons if they exist
    const addRoleBtn = document.querySelector('.add-btn[onclick="addCustomRole()"]');
    if (addRoleBtn) {
        addRoleBtn.onclick = addCustomRole;
    }

    const addSkillBtn = document.querySelector('.add-btn[onclick="addSkill()"]');
    if (addSkillBtn) {
        addSkillBtn.onclick = addSkill;
    }

    const uploadResumeBtn = document.querySelector('.upload-btn[onclick="redirectToUpload()"]');
    if (uploadResumeBtn) {
        uploadResumeBtn.onclick = redirectToUpload;
    }
});

// Resume Cross check page 

document.addEventListener('DOMContentLoaded', function() {
    // Find the button first
    const analyzeButton = document.getElementById('analyze-button'); // Use your actual button ID
    
    // Check if button exists
    if (!analyzeButton) {
        console.error('Analyze button not found in the document');
        return;
    }
    
    // Add event listener
    analyzeButton.addEventListener('click', async function() {
        const model = document.getElementById('model-select').value;
        const intensity = document.getElementById('intensity-select').value;
        
        if (!model || !intensity) {
            showValidationPopup("Please select both model and intensity.");
            return;
        }
        
        try {
            const response = await fetch('/process-resumes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, intensity }),
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Store all necessary data in sessionStorage
                sessionStorage.setItem('analysisResults', JSON.stringify(result.results));
                sessionStorage.setItem('model', model);
                sessionStorage.setItem('intensity', intensity);
                window.location.href = 'report.html';
            } else {
                showValidationPopup("Error processing resumes: " + (result.error || "Unknown error"));
            }
        } catch (error) {
            console.error('Error processing resumes:', error);
            showValidationPopup("Failed to process resumes. Please try again.");
        }
    });
});

// Report page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Only run this code if we're on the report page
    if (!document.querySelector('.report-content')) return;

    const loadingScreen = document.getElementById('loadingScreen');
    const reportContent = document.getElementById('reportContent');
    const progressBar = document.getElementById('analysisProgress');
    const resultsContainer = document.getElementById('resultsContainer');

    // Check if we have model/intensity from result.html
    const model = sessionStorage.getItem('model') || 'Skills only';
    const intensity = sessionStorage.getItem('intensity') || 'Casual';
    const analysisResults = JSON.parse(sessionStorage.getItem('analysisResults') || '[]');
    
    console.log(`Analysis started with Model: ${model}, Intensity: ${intensity}`);

    // Floating dock elements
    const floatingDock = document.querySelector('.floating-dock');
    const filterRadios = document.querySelectorAll('input[name="filter"]');
    const exportExcelBtn = document.getElementById('exportExcel');
    const exportPDFBtn = document.getElementById('exportPDF');
    let currentResults = [...analysisResults]; // Store the current results

    // Simulate or perform actual data loading
    simulateDataLoading();

    function simulateDataLoading() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 100) progress = 100;
            progressBar.style.width = `${progress}%`;
            
            if (progress === 100) {
                clearInterval(interval);
                setTimeout(() => {
                    loadingScreen.style.opacity = '0';
                    setTimeout(() => {
                        loadingScreen.style.display = 'none';
                        reportContent.style.display = 'block';
                        displayResults(currentResults);
                        initializeFloatingDock();
                    }, 500);
                }, 500);
            }
        }, 300);
    }

    function initializeFloatingDock() {
        const originalResults = [...analysisResults];
    
        filterRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                if (!originalResults.length) return;
                
                const filterValue = this.value;
                let filteredResults = [];
                
                switch(filterValue) {
                    case 'all':
                        filteredResults = [...originalResults]; // Use original results
                        break;
                    case 'top10':
                        filteredResults = [...originalResults]
                            .sort((a, b) => (b.score || 0) - (a.score || 0))
                            .slice(0, 10);
                        break;
                    case 'positive':
                        filteredResults = originalResults.filter(result => result.prediction === true);
                        break;
                    case 'negative':
                        filteredResults = originalResults.filter(result => result.prediction === false);
                        break;
                    default:
                        filteredResults = [...originalResults];
                }
                
                currentResults = filteredResults;
                displayResults(currentResults);
            });
        });
        // Export functionality
        exportExcelBtn.addEventListener('click', function() {
            if (currentResults.length === 0) {
                showValidationPopup("No data to export");
                return;
            }
            exportToExcel(currentResults);
            console.log("Exporting to Excel:", currentResults);
            showValidationPopup("Preparing Excel export...");
            setTimeout(() => {
                showValidationPopup("Excel export started. Check your downloads.");
            }, 1000);
        });
        
        exportPDFBtn.addEventListener('click', function() {
            if (currentResults.length === 0) {
                showValidationPopup("No data to export");
                return;
            }
            exportToPDF(currentResults);
            console.log("Exporting to PDF:", currentResults);
            showValidationPopup("Generating PDF report...");
            setTimeout(() => {
                showValidationPopup("PDF report generated. Check your downloads.");
            }, 1000);
        });
    }

    function displayResults(results) {
        if (results.length > 0) {
            // Display actual results if we have them
            resultsContainer.innerHTML = `
                <h2 class="section-title">Matching Results</h2>
                ${results.map(result => {
                    // Calculate additional metrics if they exist and are non-zero
                    const hasSkillSimilarity = result.skill_similarity && result.skill_similarity !== 0;
                    const hasEducationScore = result.education_score && result.education_score !== 0;
                    const hasCertificationCount = result.certification_count && result.certification_count !== 0;
                    const hasExperienceScore = result.experience_score && result.experience_score !== 0;
                    const hasKeywordMatch = result.keyword_match && result.keyword_match !== 0;
                    
                    return `
                    <div class="result-card-report">
                        <div class="card-header-report">
                            <h2 class="candidate-name-report">
                                ${result.file_name || 'Unknown File'}
                                <span class="file-type ${result.file_name?.toLowerCase().includes('.pdf') ? 'pdf' : 'docx'}">
                                    ${result.file_name?.toLowerCase().includes('.pdf') ? 'PDF' : 'DOCX'}
                                </span>
                            </h2>
                            <p class="model-info">Model: ${model} (${intensity})</p>
                        </div>
                        <div class="card-body-report">
                            <div class="contact-info-report">
                                <div class="contact-item-report">
                                    <span class="contact-label-report">Email</span>
                                    <span class="contact-value-report">${result.email || 'N/A'}</span>
                                </div>
                                <div class="contact-item-report">
                                    <span class="contact-label-report">Phone</span>
                                    <span class="contact-value-report">${result.phone || 'N/A'}</span>
                                </div>
                            </div>
                            
                            <div class="section-report">
                                <h3 class="section-title-report">Matched Skills</h3>
                                <div class="skills-container-report">
                                    ${result.matched_skills && result.matched_skills.length > 0 ? 
                                        result.matched_skills.map(skill => `<span class="skill-tag-report">${skill}</span>`).join('') :
                                        '<span class="no-skills">No skills matched</span>'}
                                </div>
                            </div>
                            
                            <div class="section-report">
                                <h3 class="section-title-report">Missing Skills</h3>
                                <div class="skills-container-report">
                                    ${result.missing_skills && result.missing_skills.length > 0 ? 
                                        result.missing_skills.map(skill => `<span class="skill-tag-report">${skill}</span>`).join('') :
                                        '<span class="no-skills">None</span>'}
                                </div>
                            </div>
                            
                            ${(hasSkillSimilarity || hasEducationScore || hasCertificationCount || hasExperienceScore || hasKeywordMatch) ? `
                            <div class="section-report">
                                <h3 class="section-title-report">Detailed Metrics</h3>
                                <div class="metrics-grid">
                                    ${hasSkillSimilarity ? `
                                    <div class="metric-item">
                                        <span class="metric-label">Skill Similarity</span>
                                        <span class="metric-value">${(result.skill_similarity * 100).toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                    
                                    ${hasEducationScore ? `
                                    <div class="metric-item">
                                        <span class="metric-label">Education Score</span>
                                        <span class="metric-value">${(result.education_score * 100).toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                    
                                    ${hasCertificationCount ? `
                                    <div class="metric-item">
                                        <span class="metric-label">Certifications</span>
                                        <span class="metric-value">${result.certification_count}</span>
                                    </div>
                                    ` : ''}
                                    
                                    ${hasExperienceScore ? `
                                    <div class="metric-item">
                                        <span class="metric-label">Experience Score</span>
                                        <span class="metric-value">${(result.experience_score * 100).toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                    
                                    ${hasKeywordMatch ? `
                                    <div class="metric-item">
                                        <span class="metric-label">Keyword Match</span>
                                        <span class="metric-value">${(result.keyword_match * 100).toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                </div>
                            </div>
                            ` : ''}
                            
                            <div class="match-info-report">
                                <div class="score-report">${result.score ? (result.score * 100).toFixed(1) + '%' : 'N/A'}</div>
                                <div class="prediction-report">
                                    <span class="prediction-label-report">Prediction:</span>
                                    <span class="prediction-value-report ${result.prediction ? 'match' : 'no-match'}">
                                        ${result.prediction ? 'Match' : 'Not a Match'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    `;
                }).join('')}
            `;
        } else {
            // Display placeholder if no results
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <h2 class="section-title">Analysis Complete</h2>
                    <p>No results to display. Model: ${model} (${intensity})</p>
                </div>
            `;
        }
    }

    async function exportToExcel(data) {
        try {
            showValidationPopup("Preparing Excel export...");
            
            // Clone and prepare the data
            const exportData = data.map(item => ({
                file_name: item.file_name || 'Unknown',
                email: item.email || 'N/A',
                phone: item.phone || 'N/A',
                score: item.score || 0,
                prediction: item.prediction,
                matched_skills: item.matched_skills || []
            }));
            
            console.log("Sending data for Excel export:", exportData.length, "records");
            
            const response = await fetch('/export/excel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                },
                body: JSON.stringify(exportData)
            });
        
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to export Excel: ${response.status} ${errorText}`);
            }
        
            const blob = await response.blob();
            if (blob.size === 0) {
                throw new Error('Received empty file from server');
            }
            
            console.log("Downloaded blob:", blob.type, blob.size, "bytes");
            
            // Create and trigger download
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'Resume-Report.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            
            showValidationPopup("Excel export completed successfully!");
        } catch (error) {
            console.error('Excel export failed:', error);
            showValidationPopup(`Excel export failed: ${error.message}`);
        }
    }
    
    async function exportToPDF(data) {
        try {
            const response = await fetch('/export/pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
    
            if (!response.ok) {
                throw new Error('Failed to export PDF');
            }
    
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'Resume-Report.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (error) {
            console.error('PDF export failed:', error);
        }
    }
    
});