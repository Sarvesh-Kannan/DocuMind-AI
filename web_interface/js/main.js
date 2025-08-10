// DocuMind AI - Main JavaScript File
// Handles frontend interactions and backend API communications

class DocuMindApp {
    constructor() {
        this.apiUrl = 'http://localhost:8080/api';
        this.isSystemInitialized = false;
        this.currentDocuments = [];
        this.currentSuggestions = [];
        this.searchResults = [];
        this.currentPage = 1;
        this.resultsPerPage = 5;
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.checkSystemStatus();
        this.updateStatusIndicator();
    }

    setupEventListeners() {
        // File upload
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const processBtn = document.getElementById('process-btn');

        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));
        
        // File selection
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Process button
        processBtn.addEventListener('click', this.processDocuments.bind(this));

        // Search functionality
        const searchBtn = document.getElementById('search-btn');
        const queryInput = document.getElementById('query-input');
        
        searchBtn.addEventListener('click', this.performSearch.bind(this));
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });

        // Pagination
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        
        if (prevBtn) prevBtn.addEventListener('click', () => this.changePage(-1));
        if (nextBtn) nextBtn.addEventListener('click', () => this.changePage(1));
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/status`);
            const data = await response.json();
            
            this.isSystemInitialized = data.initialized;
            this.updateSystemStats(data);
            this.updateStatusIndicator(data.llm_available);
            
            if (data.documents_count > 0) {
                await this.loadSuggestions();
            }
        } catch (error) {
            console.error('Failed to check system status:', error);
            this.updateStatusIndicator(false);
        }
    }

    updateSystemStats(data) {
        document.getElementById('doc-count').textContent = data.documents_count || 0;
        document.getElementById('chunk-count').textContent = data.chunks_count || 0;
        document.getElementById('response-time').textContent = data.avg_response_time || '--';
        document.getElementById('accuracy-score').textContent = data.avg_accuracy || '--';
    }

    updateStatusIndicator(isAvailable = null) {
        const indicator = document.getElementById('status-indicator');
        const dot = indicator.querySelector('div');
        const text = indicator.querySelector('span');
        
        if (isAvailable === true) {
            dot.className = 'w-3 h-3 bg-green-500 rounded-full';
            text.textContent = 'System Ready';
            text.className = 'text-sm text-green-600';
        } else if (isAvailable === false) {
            dot.className = 'w-3 h-3 bg-red-500 rounded-full';
            text.textContent = 'System Error';
            text.className = 'text-sm text-red-600';
        } else {
            dot.className = 'w-3 h-3 bg-yellow-500 rounded-full';
            text.textContent = 'Checking...';
            text.className = 'text-sm text-yellow-600';
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = 'copy';
        
        const dropZone = document.getElementById('drop-zone');
        dropZone.classList.add('border-blue-400', 'bg-blue-50');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        const dropZone = document.getElementById('drop-zone');
        dropZone.classList.remove('border-blue-400', 'bg-blue-50');
        
        const files = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
        this.displaySelectedFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.displaySelectedFiles(files);
    }

    displaySelectedFiles(files) {
        const fileList = document.getElementById('file-list');
        const processBtn = document.getElementById('process-btn');
        
        fileList.innerHTML = '';
        
        if (files.length === 0) {
            processBtn.classList.add('hidden');
            return;
        }
        
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'flex items-center justify-between bg-gray-50 p-4 rounded-xl border';
            fileItem.innerHTML = `
                <div class="flex items-center space-x-3">
                    <i class="fas fa-file-pdf text-red-500 text-xl"></i>
                    <div>
                        <div class="font-medium text-gray-900">${file.name}</div>
                        <div class="text-sm text-gray-500">${this.formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button onclick="app.removeFile(${index})" class="text-red-500 hover:text-red-700">
                    <i class="fas fa-times"></i>
                </button>
            `;
            fileList.appendChild(fileItem);
        });
        
        this.selectedFiles = files;
        processBtn.classList.remove('hidden');
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.displaySelectedFiles(this.selectedFiles);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async processDocuments() {
        if (!this.selectedFiles || this.selectedFiles.length === 0) return;
        
        const processBtn = document.getElementById('process-btn');
        const uploadProgress = document.getElementById('upload-progress');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        
        processBtn.disabled = true;
        uploadProgress.classList.remove('hidden');
        
        try {
            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            progressText.textContent = 'Uploading files...';
            progressBar.style.width = '20%';
            
            const response = await fetch(`${this.apiUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            progressText.textContent = 'Processing documents...';
            progressBar.style.width = '60%';
            
            const data = await response.json();
            
            progressText.textContent = 'Building embeddings...';
            progressBar.style.width = '90%';
            
            // Wait for embeddings to be built
            await this.waitForProcessing();
            
            progressBar.style.width = '100%';
            progressText.textContent = 'Complete!';
            
            // Update UI
            await this.checkSystemStatus();
            await this.loadSuggestions();
            
            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                this.showSuccessMessage(`Successfully processed ${this.selectedFiles.length} document(s)`);
                this.selectedFiles = [];
                document.getElementById('file-list').innerHTML = '';
                processBtn.classList.add('hidden');
            }, 1000);
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.showErrorMessage('Failed to process documents: ' + error.message);
            uploadProgress.classList.add('hidden');
        }
        
        processBtn.disabled = false;
    }

    async waitForProcessing() {
        // Poll the status endpoint until processing is complete
        for (let i = 0; i < 30; i++) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            try {
                const response = await fetch(`${this.apiUrl}/status`);
                const data = await response.json();
                if (data.processing_complete) {
                    return;
                }
            } catch (error) {
                console.warn('Status check failed:', error);
            }
        }
    }

    async loadSuggestions() {
        try {
            const response = await fetch(`${this.apiUrl}/suggestions`);
            const data = await response.json();
            
            this.currentSuggestions = data.suggestions || [];
            this.displaySuggestions();
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    }

    displaySuggestions() {
        const container = document.getElementById('suggestions-container');
        const list = document.getElementById('suggestions-list');
        
        if (this.currentSuggestions.length === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        list.innerHTML = '';
        
        this.currentSuggestions.slice(0, 8).forEach(suggestion => {
            const button = document.createElement('button');
            button.className = 'bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm hover:bg-blue-200 transition-colors';
            button.textContent = suggestion;
            button.onclick = () => {
                document.getElementById('query-input').value = suggestion;
                this.performSearch();
            };
            list.appendChild(button);
        });
    }

    async performSearch() {
        const query = document.getElementById('query-input').value.trim();
        if (!query) {
            this.showErrorMessage('Please enter a search query');
            return;
        }
        
        const searchType = document.getElementById('search-type').value;
        const topK = parseInt(document.getElementById('top-k').value);
        const summaryType = document.getElementById('summary-type').value;
        
        const loadingState = document.getElementById('loading-state');
        const resultsSection = document.getElementById('results-section');
        
        // Hide translation section for new search
        document.getElementById('translation-section').classList.add('hidden');
        document.getElementById('translation-result').classList.add('hidden');
        
        loadingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        try {
            const response = await fetch(`${this.apiUrl}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    search_type: searchType,
                    top_k: topK,
                    summary_type: summaryType
                })
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.searchResults = data.search_results || [];
            this.displayResults(data);
            
        } catch (error) {
            console.error('Search failed:', error);
            this.showErrorMessage('Search failed: ' + error.message);
        } finally {
            loadingState.classList.add('hidden');
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('results-section');
        const summaryContainer = document.getElementById('summary-container');
        const summaryContent = document.getElementById('summary-content');
        const summaryStats = document.getElementById('summary-stats');
        const searchResultsContainer = document.getElementById('search-results');
        
        resultsSection.classList.remove('hidden');
        
        // Display summary (always in English first)
        if (data.summary && data.summary.summary) {
            summaryContent.textContent = data.summary.summary;
            this.currentSummaryText = data.summary.summary; // Store for translation
            
            const stats = data.summary.summary_stats || {};
            let statsText = `${stats.word_count || 0} words • ${stats.sentence_count || 0} sentences • ${data.summary.generation_time?.toFixed(2) || 0}s`;
            summaryStats.textContent = statsText;
            
            // Show translation section
            document.getElementById('translation-section').classList.remove('hidden');
            this.setupTranslationEvents();
        }
        
        // Display search results
        this.displaySearchResults();
        
        // Update performance stats
        if (data.performance) {
            document.getElementById('response-time').textContent = `${data.performance.total_time?.toFixed(2)}s`;
            document.getElementById('accuracy-score').textContent = `${(data.performance.avg_accuracy * 100)?.toFixed(1)}%`;
        }
    }

    setupTranslationEvents() {
        const languageSelect = document.getElementById('translation-language');
        const translateBtn = document.getElementById('translate-btn');
        
        // Enable/disable translate button based on language selection
        languageSelect.addEventListener('change', () => {
            translateBtn.disabled = !languageSelect.value;
        });
        
        // Handle translation
        translateBtn.addEventListener('click', () => {
            this.translateSummary();
        });
    }

    async translateSummary() {
        const languageSelect = document.getElementById('translation-language');
        const targetLanguage = languageSelect.value;
        
        if (!targetLanguage || !this.currentSummaryText) {
            return;
        }
        
        const loadingDiv = document.getElementById('translation-loading');
        const resultDiv = document.getElementById('translation-result');
        
        // Show loading state
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        
        try {
            const response = await fetch(`${this.apiUrl}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: this.currentSummaryText,
                    target_language: targetLanguage
                })
            });
            
            if (!response.ok) {
                throw new Error(`Translation failed: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide loading
            loadingDiv.classList.add('hidden');
            
            if (data.status === 'success') {
                // Show translation result
                const langLabel = document.getElementById('translation-lang-label');
                const statusSpan = document.getElementById('translation-status');
                const textDiv = document.getElementById('translation-text');
                
                // Get language display name
                const option = languageSelect.options[languageSelect.selectedIndex];
                langLabel.textContent = option.text;
                statusSpan.textContent = '✅ Translation successful';
                textDiv.textContent = data.translated_text;
                
                resultDiv.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Translation failed');
            }
            
        } catch (error) {
            loadingDiv.classList.add('hidden');
            console.error('Translation error:', error);
            
            // Show error in result div
            const statusSpan = document.getElementById('translation-status');
            const textDiv = document.getElementById('translation-text');
            
            statusSpan.textContent = '❌ Translation failed';
            textDiv.textContent = 'Translation failed. Please try again.';
            
            resultDiv.classList.remove('hidden');
        }
    }

    displaySearchResults() {
        const container = document.getElementById('search-results');
        const resultsContainer = container.querySelector('div:last-child') || container;
        
        // Clear previous results (keep the header)
        const existingResults = container.querySelectorAll('.search-result-item');
        existingResults.forEach(item => item.remove());
        
        if (this.searchResults.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'text-center py-8 text-gray-500';
            noResults.innerHTML = '<i class="fas fa-search text-4xl mb-4"></i><p>No relevant documents found</p>';
            container.appendChild(noResults);
            return;
        }
        
        // Calculate pagination
        const startIndex = (this.currentPage - 1) * this.resultsPerPage;
        const endIndex = startIndex + this.resultsPerPage;
        const pageResults = this.searchResults.slice(startIndex, endIndex);
        
        pageResults.forEach((result, index) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item bg-gray-50 border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow';
            
            const accuracyScore = (result.accuracy_score * 100).toFixed(1);
            const accuracyColor = result.accuracy_score > 0.7 ? 'text-green-600' : result.accuracy_score > 0.5 ? 'text-yellow-600' : 'text-red-600';
            
            resultItem.innerHTML = `
                <div class="flex justify-between items-start mb-3">
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-file-pdf text-red-500"></i>
                        <span class="font-semibold text-gray-900">${result.file_name}</span>
                        <span class="text-sm text-gray-500">Page ${result.page_number}</span>
                    </div>
                    <div class="flex items-center space-x-3">
                        <div class="text-right">
                            <div class="text-sm ${accuracyColor} font-semibold">${accuracyScore}%</div>
                            <div class="text-xs text-gray-500">Accuracy</div>
                        </div>
                        ${result.hybrid_score ? `
                        <div class="text-right">
                            <div class="text-sm text-blue-600 font-semibold">${(result.hybrid_score * 100).toFixed(1)}%</div>
                            <div class="text-xs text-gray-500">Relevance</div>
                        </div>
                        ` : ''}
                    </div>
                </div>
                <div class="text-gray-700 leading-relaxed">
                    ${this.highlightQuery(result.text, document.getElementById('query-input').value)}
                </div>
                <div class="mt-3 text-xs text-gray-500">
                    Chunk ${result.chunk_id} • ${result.text.length} characters
                </div>
            `;
            
            container.appendChild(resultItem);
        });
        
        this.updatePagination();
    }

    highlightQuery(text, query) {
        if (!query) return text;
        
        const words = query.toLowerCase().split(/\s+/);
        let highlightedText = text;
        
        words.forEach(word => {
            if (word.length > 2) {
                const regex = new RegExp(`(${word})`, 'gi');
                highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
            }
        });
        
        return highlightedText;
    }

    updatePagination() {
        const pagination = document.getElementById('pagination');
        const pageInfo = document.getElementById('page-info');
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        
        if (this.searchResults.length <= this.resultsPerPage) {
            pagination.classList.add('hidden');
            return;
        }
        
        pagination.classList.remove('hidden');
        
        const totalPages = Math.ceil(this.searchResults.length / this.resultsPerPage);
        pageInfo.textContent = `Page ${this.currentPage} of ${totalPages}`;
        
        prevBtn.disabled = this.currentPage === 1;
        nextBtn.disabled = this.currentPage === totalPages;
        
        if (prevBtn.disabled) {
            prevBtn.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
        
        if (nextBtn.disabled) {
            nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    changePage(direction) {
        const totalPages = Math.ceil(this.searchResults.length / this.resultsPerPage);
        
        if (direction === -1 && this.currentPage > 1) {
            this.currentPage--;
        } else if (direction === 1 && this.currentPage < totalPages) {
            this.currentPage++;
        }
        
        this.displaySearchResults();
        
        // Scroll to results
        document.getElementById('search-results').scrollIntoView({ behavior: 'smooth' });
    }

    showSuccessMessage(message) {
        this.showNotification(message, 'success');
    }

    showErrorMessage(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-xl shadow-lg z-50 max-w-sm transform transition-all duration-300 translate-x-full`;
        
        if (type === 'success') {
            notification.className += ' bg-green-500 text-white';
            notification.innerHTML = `<i class="fas fa-check-circle mr-2"></i>${message}`;
        } else if (type === 'error') {
            notification.className += ' bg-red-500 text-white';
            notification.innerHTML = `<i class="fas fa-exclamation-circle mr-2"></i>${message}`;
        } else {
            notification.className += ' bg-blue-500 text-white';
            notification.innerHTML = `<i class="fas fa-info-circle mr-2"></i>${message}`;
        }
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
}

// Utility functions
function scrollToUpload() {
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
}

// Initialize the app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new DocuMindApp();
}); 