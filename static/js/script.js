document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const uploadMessage = document.getElementById('uploadMessage');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');

    if (!fileInput.files.length) {
        uploadMessage.innerHTML = '<div class="alert alert-error">Please select a CSV file.</div>';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        uploadMessage.innerHTML = '';
        loadingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            uploadMessage.innerHTML = `<div class="alert alert-success">${result.message}</div>`;
            pollForResults();
        } else {
            uploadMessage.innerHTML = `<div class="alert alert-error">${result.error}</div>`;
            loadingIndicator.classList.add('hidden');
        }
    } catch (error) {
        uploadMessage.innerHTML = '<div class="alert alert-error">Error uploading file.</div>';
        loadingIndicator.classList.add('hidden');
    }
});

async function pollForResults() {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsSection = document.getElementById('resultsSection');
    const loadingIndicator = document.getElementById('loadingIndicator');

    const checkResults = async () => {
        try {
            const response = await fetch('/results');
            if (response.ok) {
                const results = await response.json();
                resultsContainer.innerHTML = '';
                results.forEach((result, index) => {
                    const card = document.createElement('div');
                    card.className = 'card animate-slide-up';
                    card.style.animationDelay = `${index * 0.1}s`;
                    card.innerHTML = `
                        <img src="/output/${result.username}.jpg" alt="${result.username}" class="result-image">
                        <div class="p-6">
                            <h3 class="text-xl font-semibold text-blue-100">${result.username}</h3>
                            <p class="text-blue-200 italic mt-2">${result.caption}</p>
                            <div class="mt-4 space-y-2">
                                <div class="detection-item">
                                    <span class="font-medium">Cowboy Hat:</span>
                                    <span>${result.cowboy_hat ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">Bikini:</span>
                                    <span>${result.bikini ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">Horse:</span>
                                    <span>${result.horse ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">Tractor:</span>
                                    <span>${result.tractor ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">One Man:</span>
                                    <span>${result.one_man ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">One Woman:</span>
                                    <span>${result.one_woman ? '✅ Detected' : '❌ Not Detected'}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">Face Count:</span>
                                    <span>${result.face_count}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">Hair Color:</span>
                                    <span>${result.hair_color.charAt(0).toUpperCase() + result.hair_color.slice(1)}</span>
                                </div>
                                <div class="detection-item">
                                    <span class="font-medium">NSFW Category:</span>
                                    <span>${result.nsfw_category}</span>
                                </div>
                            </div>
                        </div>
                    `;
                    resultsContainer.appendChild(card);
                });
                resultsSection.classList.remove('hidden');
                loadingIndicator.classList.add('hidden');
            } else {
                setTimeout(checkResults, 2000); // Poll every 2 seconds
            }
        } catch (error) {
            setTimeout(checkResults, 2000);
        }
    };

    checkResults();
}