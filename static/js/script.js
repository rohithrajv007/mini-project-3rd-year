document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropArea = document.getElementById('dropArea');
    const audioInput = document.getElementById('audioFile');
    const fileInfo = document.getElementById('fileInfo');
    const submitBtn = document.getElementById('submitBtn');
    const paddingSlider = document.getElementById('padding');
    const paddingValue = document.getElementById('paddingValue');
    const uploadForm = document.getElementById('uploadForm');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Handle file selection via the file input
    audioInput.addEventListener('change', handleFiles);

    // Handle padding slider change
    paddingSlider.addEventListener('input', updatePaddingValue);

    // Click on drop area to trigger file input
    dropArea.addEventListener('click', function() {
        audioInput.click();
    });

    // Handle form submission with loading state
    uploadForm.addEventListener('submit', function() {
        if (submitBtn.disabled) return false;
        
        // Change button text and disable
        submitBtn.textContent = 'Processing...';
        submitBtn.disabled = true;
        
        // Add a loading class to the form
        uploadForm.classList.add('loading');
        
        // Submit the form (the default behavior continues)
        return true;
    });

    // Prevent default drag behavior
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area
    function highlight() {
        dropArea.classList.add('dragover');
    }

    // Remove highlight
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    // Handle dropped files
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Process the files
    function handleFiles(e) {
        let files;
        if (e.dataTransfer) {
            files = e.dataTransfer.files;
        } else if (e.target && e.target.files) {
            files = e.target.files;
        } else {
            files = e;
        }

        if (files.length > 0) {
            const file = files[0];
            
            // Check if it's an audio file
            if (!file.type.match('audio.*')) {
                fileInfo.textContent = 'Please select an audio file';
                fileInfo.style.color = 'red';
                submitBtn.disabled = true;
                return;
            }
            
            // Update the file info
            fileInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
            fileInfo.style.color = 'var(--text-secondary)';
            
            // Enable the submit button
            submitBtn.disabled = false;
            
            // Set the file to the input
            audioInput.files = files;
        }
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1048576) {
            return (bytes / 1024).toFixed(2) + ' KB';
        } else {
            return (bytes / 1048576).toFixed(2) + ' MB';
        }
    }

    // Update padding value display
    function updatePaddingValue() {
        paddingValue.textContent = `${paddingSlider.value} ms`;
    }

    // Initialize padding value display
    updatePaddingValue();
});

// Add pulse animation to the wave
document.addEventListener('DOMContentLoaded', function() {
    const wave = document.querySelector('.wave');
    
    // Add duplicate wave for seamless animation
    if (wave) {
        const waveClone = wave.cloneNode(true);
        waveClone.setAttribute('transform', 'translate(1440, 0)');
        wave.parentNode.appendChild(waveClone);
    }
});