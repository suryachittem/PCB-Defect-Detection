document.addEventListener('DOMContentLoaded', () => {
    // MODIFIED: Removed variables for the reference image input and preview
    const testInput = document.getElementById('test-input');
    const testPreview = document.getElementById('test-preview');
    const detectBtn = document.getElementById('detect-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const annotatedImage = document.getElementById('annotated-image');
    const predictionsList = document.getElementById('predictions-list');
    const downloadBtn = document.getElementById('download-btn');

    // This preview function remains the same but will only be used for the test image
    const setupPreview = (input, preview) => {
        input.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    };

    // MODIFIED: Removed the call for the reference input
    setupPreview(testInput, testPreview);

    detectBtn.addEventListener('click', async () => {
        // MODIFIED: Only get the test file
        const testFile = testInput.files[0];

        // MODIFIED: Updated the check to only look for the test image
        if (!testFile) {
            alert('Please upload a test image.');
            return;
        }

        // --- The UI logic for showing the loader remains the same ---
        detectBtn.disabled = true;
        detectBtn.textContent = 'Processing...';
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        annotatedImage.style.display = 'none';
        downloadBtn.classList.add('hidden');
        predictionsList.innerHTML = '';

        const formData = new FormData();
        // MODIFIED: Only append the test_image. The ref_image is removed.
        formData.append('test_image', testFile);

        try {
            // The fetch request is the same URL, but now sends less data
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'An unknown error occurred.');
            displayResults(result);
        } catch (error) {
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
            console.error('Prediction failed:', error);
        } finally {
            loader.classList.add('hidden');
            detectBtn.disabled = false;
            detectBtn.textContent = '🚀 Detect Defects';
        }
    });

    // --- The displayResults function does not need any changes ---
    function displayResults(data) {
        annotatedImage.src = data.annotated_image;
        annotatedImage.style.display = 'block';

        predictionsList.innerHTML = '';
        if (data.predictions && data.predictions.length > 0) {
            data.predictions.forEach(pred => {
                const li = document.createElement('li');
                // A small text improvement for clarity in the UI
                if (pred.label === 'No Defects Found') {
                    li.textContent = pred.label;
                } else {
                    li.textContent = `Defect: ${pred.label} (Confidence: ${pred.confidence})`;
                }
                predictionsList.appendChild(li);
            });
        } else {
            predictionsList.innerHTML = '<li>Could not determine defects.</li>';
        }

        // Show download button below predictions, but only if an image was processed
        if (data.annotated_image) {
            downloadBtn.classList.remove('hidden');
            downloadBtn.onclick = () => {
                const link = document.createElement('a');
                link.href = annotatedImage.src;
                link.download = 'annotated_result.png';
                link.click();
            };
        }
    }
});