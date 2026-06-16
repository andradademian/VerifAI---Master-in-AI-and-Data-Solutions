/* ============================================================
   VerifAI — UI layer controller
   Tabs, mock auth, freemium usage-gating, subscribe flow,
   sample newsletter, and premium analysis augmentation.

   DEMO MODE: all account state is mock and lives in localStorage.
   No backend, no database, no real emails. Removing this file
   leaves the original article-scoring MVP fully functional.
   ============================================================ */
(function () {
    'use strict';

    var FREE_CHECK_LIMIT = 5;          // free article checks / month (per business plan)
    var STORAGE_KEY = 'verifai_demo_state';

    // ---------- state ----------
    var defaultState = {
        loggedIn: false,
        email: '',
        plan: 'free',                  // 'free' | 'premium'
        checksUsed: 0,
        subscription: null             // { topics, keywords, frequency, email }
    };

    function loadState() {
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if (raw) return Object.assign({}, defaultState, JSON.parse(raw));
        } catch (e) { /* ignore */ }
        return Object.assign({}, defaultState);
    }
    function saveState() {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {}
    }
    var state = loadState();

    // ---------- helpers ----------
    function $(id) { return document.getElementById(id); }
    function qsa(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }

    function toast(msg) {
        var t = $('toast');
        if (!t) return;
        t.textContent = msg;
        t.classList.add('show');
        clearTimeout(toast._t);
        toast._t = setTimeout(function () { t.classList.remove('show'); }, 2600);
    }

    function scoreClass(pct) {
        if (pct >= 70) return 'high';
        if (pct >= 40) return 'medium';
        return 'low';
    }

    // ---------- tab routing ----------
    var TABS = ['explore', 'how', 'pricing', 'subscribe'];
    function showTab(name, skipHash) {
        if (TABS.indexOf(name) === -1) return;
        qsa('.tab-pane').forEach(function (p) { p.classList.remove('active'); });
        var pane = $('tab-' + name);
        if (pane) pane.classList.add('active');
        qsa('.nav-tab').forEach(function (b) {
            b.classList.toggle('active', b.getAttribute('data-tab') === name);
        });
        if (!skipHash && window.location.hash !== '#' + name) {
            window.location.hash = name;
        }
        window.scrollTo(0, 0);
    }
    function tabFromHash() {
        var h = (window.location.hash || '').replace('#', '');
        return TABS.indexOf(h) !== -1 ? h : null;
    }

    // ---------- account UI ----------
    function renderAccount() {
        var outEl = $('accountLoggedOut');
        var inEl = $('accountLoggedIn');
        if (state.loggedIn) {
            outEl.style.display = 'none';
            inEl.style.display = 'flex';
            $('accountEmail').textContent = state.email;
            $('accountAvatar').textContent = (state.email[0] || 'U').toUpperCase();
            var badge = $('planBadge');
            badge.textContent = state.plan === 'premium' ? 'Premium' : 'Free';
            badge.classList.toggle('premium', state.plan === 'premium');

            var pill = $('usagePill');
            if (state.plan === 'premium') {
                pill.textContent = 'Unlimited checks';
            } else {
                var left = Math.max(0, FREE_CHECK_LIMIT - state.checksUsed);
                pill.textContent = left + ' / ' + FREE_CHECK_LIMIT + ' free checks left';
            }
        } else {
            outEl.style.display = 'flex';
            inEl.style.display = 'none';
        }
    }

    // ---------- auth modal (mock) ----------
    var authMode = 'login'; // 'login' | 'signup'
    function openAuth(mode) {
        authMode = mode;
        var signup = mode === 'signup';
        $('authTitle').textContent = signup ? 'Create your free account' : 'Welcome back';
        $('authSubtitle').textContent = signup
            ? 'Sign up free to score articles and get your digest.'
            : 'Log in to score articles and manage your digest.';
        $('authSubmit').textContent = signup ? 'Sign up free' : 'Log in';
        $('authSwitchText').textContent = signup ? 'Already have an account?' : 'New to VerifAI?';
        $('authSwitchBtn').textContent = signup ? 'Log in' : 'Sign up free';
        $('authError').textContent = '';
        $('authModal').classList.add('active');
        setTimeout(function () { $('authEmail').focus(); }, 50);
    }
    function closeAuth() { $('authModal').classList.remove('active'); }

    function handleAuthSubmit(e) {
        e.preventDefault();
        var email = $('authEmail').value.trim();
        if (!email || email.indexOf('@') === -1) {
            $('authError').textContent = 'Please enter a valid email address.';
            return;
        }
        state.loggedIn = true;
        state.email = email;
        if (!state.plan) state.plan = 'free';
        saveState();
        renderAccount();
        closeAuth();
        toast(authMode === 'signup' ? 'Welcome to VerifAI — you\'re on the Free plan!' : 'Logged in');
    }

    // ---------- upgrade / checkout (mock) ----------
    function openUpgrade(reason) {
        $('upgradeReason').textContent = reason ||
            'Unlock unlimited checks and the full five-dimension trust breakdown.';
        $('upgradeModal').classList.add('active');
    }
    function closeUpgrade() { $('upgradeModal').classList.remove('active'); }

    function confirmUpgrade() {
        if (!state.loggedIn) {
            // make sure there's an account to attach premium to
            state.loggedIn = true;
            state.email = state.email || 'you@example.com';
        }
        state.plan = 'premium';
        saveState();
        renderAccount();
        closeUpgrade();
        toast('🎉 You\'re Premium! Unlimited checks unlocked.');
        // refresh any open analysis with the full breakdown
        if (window.VerifAI._lastAnalysis) {
            window.VerifAI.augmentAnalysis(window.VerifAI._lastAnalysis);
        }
    }

    // ---------- freemium gating (called by script.js) ----------
    function canAnalyze() {
        if (state.plan === 'premium') return true;
        return state.checksUsed < FREE_CHECK_LIMIT;
    }
    function recordCheck() {
        if (state.plan === 'premium') return;
        state.checksUsed += 1;
        saveState();
        renderAccount();
    }
    function gateAnalyze() {
        // returns true if allowed to proceed
        if (!state.loggedIn) {
            openAuth('signup');
            toast('Sign up free to score articles');
            return false;
        }
        if (!canAnalyze()) {
            openUpgrade('You\'ve used all ' + FREE_CHECK_LIMIT +
                ' free checks this month. Upgrade for unlimited scoring.');
            return false;
        }
        return true;
    }

    // ---------- premium 5-dimension breakdown ----------
    var DIMENSIONS = [
        'Source reliability',
        'Factual accuracy',
        'Tone & framing',
        'Corroboration',
        'Transparency'
    ];
    // deterministic pseudo-scores derived from the overall credibility,
    // so the same article always shows the same breakdown.
    function dimensionScores(analysis) {
        var base = Math.round((analysis.prob_real || 0) * 100);
        var seedStr = (analysis.article_id || '') + base;
        var seed = 0;
        for (var i = 0; i < seedStr.length; i++) seed = (seed * 31 + seedStr.charCodeAt(i)) % 100000;
        return DIMENSIONS.map(function (name, idx) {
            var jitter = ((seed >> (idx * 3)) % 24) - 12;   // -12..+11
            var val = Math.max(5, Math.min(98, base + jitter));
            return { name: name, value: val };
        });
    }

    function augmentAnalysis(analysis) {
        window.VerifAI._lastAnalysis = analysis;
        var content = $('analysisContent');
        if (!content) return;
        // remove any previously injected premium block
        var existing = content.querySelector('.premium-block');
        if (existing) existing.parentNode.removeChild(existing);

        var wrap = document.createElement('div');
        wrap.className = 'analysis-section premium-block';

        if (state.plan === 'premium') {
            var rows = dimensionScores(analysis).map(function (d) {
                return '<div class="dimension-row">' +
                    '<span class="dimension-label">' + d.name + '</span>' +
                    '<span class="dimension-track"><span class="dimension-bar ' + scoreClass(d.value) +
                        '" data-w="' + d.value + '" style="width:0%"></span></span>' +
                    '<span class="dimension-val">' + d.value + '%</span>' +
                    '</div>';
            }).join('');
            wrap.innerHTML = '<h3>Five-dimension trust breakdown <span class="premium-tag">PREMIUM</span></h3>' + rows;
        } else {
            wrap.innerHTML =
                '<div class="premium-locked">' +
                '<div class="lock-icon">🔒</div>' +
                '<h4>Five-dimension trust breakdown</h4>' +
                '<p>See exactly why this article scored the way it did — source reliability, factual accuracy, tone, corroboration and transparency.</p>' +
                '<button class="btn-solid gold" id="unlockBreakdownBtn">Unlock with Premium</button>' +
                '</div>';
        }
        content.appendChild(wrap);

        // animate each bar from 0 to its percentage on next frame
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                qsa('.dimension-bar').forEach(function (b) {
                    b.style.width = (b.getAttribute('data-w') || 0) + '%';
                });
            });
        });

        var unlockBtn = $('unlockBreakdownBtn');
        if (unlockBtn) unlockBtn.addEventListener('click', function () {
            openUpgrade('Upgrade to see the full five-dimension breakdown for every article.');
        });
    }

    // ---------- sample newsletter ----------
    var SAMPLE_ARTICLES = [
        { score: 91, title: 'Central bank holds rates steady amid cooling inflation', src: 'Reuters' },
        { score: 78, title: 'New study links sleep regularity to long-term heart health', src: 'AP News' },
        { score: 64, title: 'Tech giants pledge new AI safety commitments at summit', src: 'The Verge' },
        { score: 38, title: '“Miracle” supplement claims spark fresh expert pushback', src: 'HealthDaily' }
    ];
    function newsletterHTML(opts) {
        opts = opts || {};
        var topics = opts.topics && opts.topics.length ? opts.topics.join(' · ') : 'Top stories';
        var freq = opts.frequency || 'Weekly';
        var premium = state.plan === 'premium';
        var items = SAMPLE_ARTICLES.slice(0, premium ? 4 : 3).map(function (a) {
            return '<div class="nl-item">' +
                '<div class="nl-score ' + scoreClass(a.score) + '">' + a.score + '<small>TRUST</small></div>' +
                '<div class="nl-text"><h4>' + a.title + '</h4><span class="nl-src">' + a.src + '</span></div>' +
                '</div>';
        }).join('');
        var locked = premium ? '' :
            '<div class="nl-locked">+ Unlimited scored articles &amp; the full breakdown with Premium</div>';
        return '<div class="nl-head"><div class="nl-logo">VerifAI</div>' +
            '<div class="nl-tag">' + freq + ' Trust Digest · ' + topics + '</div></div>' +
            '<div class="nl-body">' + items + '</div>' + locked +
            '<div class="nl-foot">Each score is generated by the VerifAI trust engine. Tap any story to read the full analysis.</div>';
    }
    function renderNewsletterPreviews() {
        var sub = state.subscription || {};
        var el = $('newsletterPreview');
        if (el) el.innerHTML = newsletterHTML(sub);
    }

    // ---------- subscribe flow ----------
    function handleSubscribe(e) {
        e.preventDefault();
        var topics = qsa('#topicGrid input:checked').map(function (c) { return c.value; });
        var keywords = $('keywordsInput').value.trim();
        var freqEl = document.querySelector('input[name="freq"]:checked');
        var frequency = freqEl ? freqEl.value : 'Weekly';
        var email = $('subscribeEmail').value.trim();
        var err = $('subscribeError');
        err.textContent = '';

        if (topics.length === 0) { err.textContent = 'Please pick at least one topic.'; return; }
        if (!email || email.indexOf('@') === -1) { err.textContent = 'Please enter a valid email address.'; return; }

        state.subscription = { topics: topics, keywords: keywords, frequency: frequency, email: email };
        // subscribing also creates a free account if needed
        if (!state.loggedIn) { state.loggedIn = true; state.email = email; state.plan = state.plan || 'free'; }
        saveState();
        renderAccount();

        // summary + preview
        var parts = [frequency + ' digest', topics.join(', ')];
        if (keywords) parts.push('keywords: ' + keywords);
        $('subscribeSummary').textContent = parts.join(' · ') + ' → ' + email;
        $('successPreview').innerHTML = newsletterHTML(state.subscription);
        $('subscribeForm').style.display = 'none';
        $('subscribeSuccess').style.display = 'block';
        window.scrollTo(0, 0);
        toast('Subscribed! Check the preview below.');
    }

    // ---------- wire up ----------
    function init() {
        // tab buttons (any element with data-tab)
        qsa('[data-tab]').forEach(function (el) {
            el.addEventListener('click', function (ev) {
                var name = el.getAttribute('data-tab');
                if (!name) return;
                // close account menu if open
                $('accountMenu') && $('accountMenu').classList.remove('open');
                showTab(name);
            });
        });

        // auth
        $('navLoginBtn').addEventListener('click', function () { openAuth('login'); });
        $('navSignupBtn').addEventListener('click', function () { openAuth('signup'); });
        $('authForm').addEventListener('submit', handleAuthSubmit);
        $('authSwitchBtn').addEventListener('click', function () { openAuth(authMode === 'login' ? 'signup' : 'login'); });

        // account chip menu
        var chip = $('accountChip');
        if (chip) chip.addEventListener('click', function (ev) {
            if (ev.target.closest('.account-menu')) return; // let menu buttons handle themselves
            $('accountMenu').classList.toggle('open');
            ev.stopPropagation();
        });
        document.addEventListener('click', function () {
            var m = $('accountMenu'); if (m) m.classList.remove('open');
        });
        $('logoutBtn').addEventListener('click', function () {
            state.loggedIn = false;
            saveState();
            renderAccount();
            toast('Logged out');
        });

        // upgrade
        $('confirmUpgradeBtn').addEventListener('click', confirmUpgrade);

        // pricing actions
        qsa('[data-action]').forEach(function (el) {
            el.addEventListener('click', function () {
                var a = el.getAttribute('data-action');
                if (a === 'upgrade') {
                    openUpgrade();
                } else if (a === 'start-free') {
                    if (state.loggedIn) { showTab('subscribe'); }
                    else { openAuth('signup'); }
                }
            });
        });

        // billing toggle
        var toggle = $('billingToggle');
        if (toggle) toggle.addEventListener('change', function () {
            if (toggle.checked) {
                $('premiumPrice').textContent = '$49';
                $('premiumPer').textContent = '/year';
                $('premiumNote').textContent = 'Billed annually · ~$4.08/mo · cancel anytime';
            } else {
                $('premiumPrice').textContent = '$5.99';
                $('premiumPer').textContent = '/month';
                $('premiumNote').textContent = 'Billed monthly · cancel anytime';
            }
        });

        // close buttons for generic modals
        qsa('[data-close]').forEach(function (el) {
            el.addEventListener('click', function () {
                var which = el.getAttribute('data-close');
                if (which === 'auth') closeAuth();
                if (which === 'upgrade') closeUpgrade();
            });
        });
        // click outside generic modal closes it
        qsa('.va-modal').forEach(function (m) {
            m.addEventListener('click', function (ev) { if (ev.target === m) m.classList.remove('active'); });
        });

        // subscribe form
        $('subscribeForm').addEventListener('submit', handleSubscribe);

        renderAccount();
        renderNewsletterPreviews();

        // deep-link / back-button support for tabs (#pricing, #subscribe, ...)
        window.addEventListener('hashchange', function () {
            var t = tabFromHash();
            if (t) showTab(t, true);
        });
        var initial = tabFromHash();
        if (initial) showTab(initial, true);
    }

    // ---------- public API (used by script.js) ----------
    window.VerifAI = {
        gateAnalyze: gateAnalyze,
        canAnalyze: canAnalyze,
        recordCheck: recordCheck,
        isPremium: function () { return state.plan === 'premium'; },
        isLoggedIn: function () { return state.loggedIn; },
        augmentAnalysis: augmentAnalysis,
        showTab: showTab,
        _lastAnalysis: null
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
