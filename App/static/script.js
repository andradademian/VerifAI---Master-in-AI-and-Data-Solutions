let articles = [];
let nextPage = null;
let isLoading = false;
let currentCategory = '';
let currentSearchQuery = '';

window.addEventListener('DOMContentLoaded', () => {
    loadArticles();
    setupEventListeners();
    setupInfiniteScroll();
});

function setupEventListeners() {
    document.getElementById('refreshBtn').addEventListener('click', () => {
        articles = [];
        nextPage = null;
        loadArticles();
    });
    document.getElementById('categorySelect').addEventListener('change', () => {
        articles = [];
        nextPage = null;
        loadArticles();
    });
    document.getElementById('searchBtn').addEventListener('click', () => {
        articles = [];
        nextPage = null;
        loadArticles();
    });
    document.querySelector('.modal-close').addEventListener('click', closeModal);

    // Allow Enter key to search
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            articles = [];
            nextPage = null;
            loadArticles();
        }
    });

    // Close modal when clicking outside
    document.getElementById('analysisModal').addEventListener('click', (e) => {
        if (e.target.id === 'analysisModal') {
            closeModal();
        }
    });
}

function setupInfiniteScroll() {
    window.addEventListener('scroll', () => {
        // Check if user scrolled near bottom
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 500) {
            if (nextPage && !isLoading) {
                loadMoreArticles();
            }
        }
    });
}

async function loadArticles(append = false) {
    const category = document.getElementById('categorySelect').value;
    const searchQuery = document.getElementById('searchInput').value.trim();
    const loading = document.getElementById('loading');
    const container = document.getElementById('articlesContainer');
    const noArticles = document.getElementById('noArticles');
    const refreshBtn = document.getElementById('refreshBtn');
    const searchBtn = document.getElementById('searchBtn');

    if (!append) {
        isLoading = true;
        loading.style.display = 'block';
        container.innerHTML = '';
        noArticles.style.display = 'none';
        refreshBtn.disabled = true;
        searchBtn.disabled = true;
    }

    currentCategory = category;
    currentSearchQuery = searchQuery;

    try {
        const params = new URLSearchParams({
            country: 'us',
            language: 'en'
        });

        if (category) {
            params.append('category', category);
        }

        if (searchQuery) {
            params.append('q', searchQuery);
        }

        if (append && nextPage) {
            params.append('page', nextPage);
        }

        const response = await fetch(`/api/articles?${params}`);
        const data = await response.json();

        if (data.status === 'success' && data.articles.length > 0) {
            // Append or replace articles
            if (append) {
                articles = articles.concat(data.articles);
            } else {
                articles = data.articles;
            }

            nextPage = data.nextPage || null;

            // Sort articles: title matches first, then body matches
            if (searchQuery) {
                articles = sortArticlesByKeyword(articles, searchQuery);
            }

            if (!append) {
                displayArticles(articles);
            } else {
                appendArticles(data.articles);
            }

            let countText = `${articles.length} article${articles.length !== 1 ? 's' : ''} loaded`;
            if (searchQuery) {
                countText += ` for "${searchQuery}"`;
            }
            if (nextPage) {
                countText += ' (scroll for more)';
            }
            document.getElementById('articleCount').textContent = countText;
        } else {
            if (!append) {
                noArticles.style.display = 'block';
                if (searchQuery) {
                    noArticles.textContent = `No articles found for "${searchQuery}". Try a different search term.`;
                } else {
                    noArticles.textContent = 'No articles found.';
                }
            }
        }
    } catch (error) {
        console.error('Error loading articles:', error);
        if (!append) {
            container.innerHTML = '<div class="no-articles">Error loading articles. Please try again.</div>';
        }
    } finally {
        if (!append) {
            loading.style.display = 'none';
            refreshBtn.disabled = false;
            searchBtn.disabled = false;
        }
        isLoading = false;
    }
}

async function loadMoreArticles() {
    if (!nextPage || isLoading) return;

    isLoading = true;
    console.log('Loading more articles...');
    await loadArticles(true);
}

function sortArticlesByKeyword(articles, keyword) {
    const lowerKeyword = keyword.toLowerCase();

    // Separate articles into title matches and body matches
    const titleMatches = [];
    const bodyMatches = [];
    const noMatches = [];

    articles.forEach(article => {
        const titleLower = (article.title || '').toLowerCase();
        const descriptionLower = (article.description || '').toLowerCase();
        const contentLower = (article.content || '').toLowerCase();

        if (titleLower.includes(lowerKeyword)) {
            titleMatches.push(article);
        } else if (descriptionLower.includes(lowerKeyword) || contentLower.includes(lowerKeyword)) {
            bodyMatches.push(article);
        } else {
            noMatches.push(article);
        }
    });

    // Return title matches first, then body matches, then others
    return [...titleMatches, ...bodyMatches, ...noMatches];
}

function displayArticles(articles) {
    const container = document.getElementById('articlesContainer');
    container.innerHTML = '';
    appendArticles(articles);
}

function appendArticles(newArticles) {
    const container = document.getElementById('articlesContainer');

    newArticles.forEach((article, globalIndex) => {
        // Calculate actual index in the full articles array
        const articleIndex = articles.findIndex(a => a.id === article.id);

        const card = document.createElement('div');
        card.className = 'article-card';

        const categories = Array.isArray(article.category) ? article.category : [];
        const categoryBadges = categories.map(cat =>
            `<span class="category-badge">${cat}</span>`
        ).join('');

        // Limit description length
        const maxDescriptionLength = 400;
        let description = article.description || 'No description available';
        if (description.length > maxDescriptionLength) {
            description = description.substring(0, maxDescriptionLength) + '...';
        }

        card.innerHTML = `
            <div class="article-meta">
                <span class="article-source">${article.source_name || article.source}</span>
                <span>•</span>
                <span>${new Date(article.published_at).toLocaleDateString()}</span>
            </div>
            ${categoryBadges}
            <h2 class="article-title">${article.title}</h2>
            <p class="article-description">${description}</p>
            <div class="article-actions">
                <button class="btn-analyze" data-index="${articleIndex}">
                    Analyze Article
                </button>
                <a href="${article.url}" target="_blank" class="btn-read">
                    Read Full Article
                </a>
            </div>
        `;

        // Add event listener to the analyze button
        const analyzeBtn = card.querySelector('.btn-analyze');
        analyzeBtn.addEventListener('click', () => analyzeArticle(articleIndex));

        container.appendChild(card);
    });
}

async function analyzeArticle(index) {
    const article = articles[index];
    const modal = document.getElementById('analysisModal');

    // Freemium gating (demo layer): require sign-up + enforce the free
    // monthly check limit. No-op if the UI layer (ui.js) is absent.
    if (window.VerifAI && !window.VerifAI.gateAnalyze()) {
        return;
    }

    // Show modal with loading state
    showModal();
    document.getElementById('analysisContent').innerHTML = '<div class="loading">Analyzing article...</div>';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: article.title,
                text: article.content,
                article_id: article.id
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Count this successful check against the free quota (demo layer)
            if (window.VerifAI) {
                window.VerifAI.recordCheck();
                data.analysis.article_id = data.article_id || article.id || '';
            }
            displayAnalysis(data.analysis);
        } else {
            document.getElementById('analysisContent').innerHTML =
                `<div class="no-articles">Error: ${data.message}</div>`;
        }
    } catch (error) {
        console.error('Error analyzing article:', error);
        document.getElementById('analysisContent').innerHTML =
            '<div class="no-articles">Error analyzing article. Please try again.</div>';
    }
}

function displayAnalysis(analysis) {
    const probReal = Math.round(analysis.prob_real * 100);
    const probFake = Math.round(analysis.prob_fake * 100);

    let credibilityClass = 'high';
    if (probReal < 40) credibilityClass = 'low';
    else if (probReal < 70) credibilityClass = 'medium';

    // AI recommendation section (only shown when Cohere returned something)
    let aiSection = '';
    if (analysis.ai_recommendation) {
        const recHtml = escapeHtml(analysis.ai_recommendation).replace(/\n/g, '<br>');
        aiSection = `
        <div class="analysis-section ai-recommendation">
            <h3>AI Explanation</h3>
            <p>${recHtml}</p>
            <p class="ai-source">Generated by Cohere</p>
        </div>`;
    } else {
        aiSection = `
        <div class="analysis-section ai-recommendation">
            <h3>AI Explanation</h3>
            <p class="ai-source">Unavailable — set COHERE_API_KEY in your .env to enable AI recommendations.</p>
        </div>`;
    }

    const demoBanner = analysis.demo_mode
        ? `<div class="demo-banner">⚠️ Demo mode — model weights not installed, so the score below is a placeholder. The AI message is still generated live by Cohere.</div>`
        : '';

    document.getElementById('analysisContent').innerHTML = `
        <h2>Analysis Results</h2>
        ${demoBanner}

        <div class="analysis-section">
            <h3>Credibility Score</h3>
            <div class="credibility-bar">
                <div class="credibility-fill ${credibilityClass}" style="width: ${probReal}%">
                    ${probReal}%
                </div>
            </div>
            <p><strong>Real:</strong> ${probReal}%</p>
            <p><strong>Fake:</strong> ${probFake}%</p>
            <p><strong>Classification:</strong> ${analysis.classification}</p>
        </div>
        ${aiSection}
    `;

    // Inject the premium five-dimension breakdown (or its locked upsell)
    if (window.VerifAI) {
        window.VerifAI.augmentAnalysis(analysis);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showModal() {
    document.getElementById('analysisModal').classList.add('active');
}

function closeModal() {
    document.getElementById('analysisModal').classList.remove('active');
}