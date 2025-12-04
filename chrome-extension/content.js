(async () => {
    console.log("[FactCheck] Extension script started");
    // 저장된 API URL 불러오기 (기본값: http://localhost:8000)
    const getApiUrl = () => {
        return new Promise((resolve) => {
            chrome.storage.sync.get({
                apiUrl: 'http://localhost:8000'
            }, (items) => {
                console.log("[FactCheck] Using API URL:", items.apiUrl);
                resolve(items.apiUrl);
            });
        });
    };

    const API_BASE_URL = await getApiUrl();

    // 1. UI 삽입 위치 선정 (네이버 뉴스 본문 상단)
    // 네이버 뉴스 구조에 따라 선택자가 다를 수 있음. 여러 가지 시도.
    const articleBody = document.querySelector('#dic_area') || document.querySelector('#articeBody') || document.querySelector('#newsEndContents');
    const titleElement = document.querySelector('.media_end_head_title') || document.querySelector('#articleTitle') || document.querySelector('.end_tit');
    
    // 본문이나 제목이 없으면 실행하지 않음 (뉴스 페이지가 아닐 수 있음)
    if (!articleBody || !titleElement) {
        console.log("[FactCheck] Not a news article page (elements missing).");
        return;
    }

    console.log("[FactCheck] Ready to request.");

    try {
        // 3. API 호출
        const currentUrl = window.location.href;
        console.log("[FactCheck] Requesting analysis for:", currentUrl);

        // API 서버에 현재 URL을 보내서 분석 요청
        const response = await fetch(`${API_BASE_URL}/check-facts?url=${encodeURIComponent(currentUrl)}`);
        
        console.log("[FactCheck] Response status:", response.status);

        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        console.log("[FactCheck] Data received:", data);
        
        // 4. 결과 렌더링 (거짓인 정보가 1개 이상일 때만 표시)
        const hasFakeNews = data.related_factchecks && data.related_factchecks.some(item => {
            const judgment = item.verification?.judgment || '';
            return judgment.includes('거짓');
        });

        if (hasFakeNews) {
            console.log("[FactCheck] Fake news detected. Injecting UI.");
            
            // 컨테이너 생성 및 삽입
            const container = document.createElement('div');
            container.className = 'factcheck-container';
            titleElement.after(container);
            
            renderResult(container, data);
        } else {
            console.log("[FactCheck] No fake news detected. UI skipped.");
        }

    } catch (error) {
        console.error("FactCheck Extension Error:", error);
        // 에러 발생 시 UI를 표시하지 않음
    }
})();

function renderResult(container, data) {
    console.log("[FactCheck] Rendering result UI");
    const { related_factchecks } = data;

    // 분류: 거짓(Fake) vs 그 외(True/Unknown)
    const fakeItems = [];
    const otherItems = [];

    related_factchecks.forEach(item => {
        const judgment = item.verification?.judgment || '판단 불가';
        if (judgment.includes('거짓')) {
            fakeItems.push(item);
        } else {
            otherItems.push(item);
        }
    });

    // 결과 HTML 조립
    let html = `
        <div class="factcheck-header">
            <span class="factcheck-logo">🤖 FactCheck AI</span>
            <span class="factcheck-title" style="color: #ff4b4b;">⚠️ 팩트체크 경고</span>
        </div>
    `;

    // 1. 거짓 정보 (항상 표시)
    fakeItems.forEach(item => {
        html += createItemHtml(item, 'fake');
    });

    // 2. 그 외 정보 (자세히 보기로 숨김)
    if (otherItems.length > 0) {
        html += `
            <button id="factcheck-toggle-btn" class="factcheck-details-toggle">
                참/판단불가 정보 자세히 보기 (${otherItems.length}건) ▼
            </button>
            <div id="factcheck-details" class="factcheck-details-container">
        `;
        
        otherItems.forEach(item => {
            const judgment = item.verification?.judgment || '판단 불가';
            let type = 'unknown';
            if (judgment.includes('사실')) type = 'fact';
            
            html += createItemHtml(item, type);
        });

        html += `</div>`;
    }

    container.innerHTML = html;

    // 토글 이벤트 리스너 추가
    const toggleBtn = container.querySelector('#factcheck-toggle-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const details = container.querySelector('#factcheck-details');
            const isHidden = getComputedStyle(details).display === 'none';
            
            if (isHidden) {
                details.style.display = 'block';
                toggleBtn.textContent = `참/판단불가 정보 접기 ▲`;
            } else {
                details.style.display = 'none';
                toggleBtn.textContent = `참/판단불가 정보 자세히 보기 (${otherItems.length}건) ▼`;
            }
        });
    }
}

function createItemHtml(item, type) {
    const verification = item.verification || {};
    const judgment = verification.judgment || '판단 불가';
    const reason = verification.reason || '근거 데이터 부족';
    
    let typeClass = 'unknown';
    let badgeClass = 'badge-unknown';
    
    if (type === 'fact') {
        typeClass = 'fact';
        badgeClass = 'badge-fact';
    } else if (type === 'fake') {
        typeClass = 'fake';
        badgeClass = 'badge-fake';
    }

    // 참고 기사 링크 생성
    let referencesHtml = '';
    if (item.related_facts && item.related_facts.length > 0) {
        referencesHtml = '<div class="factcheck-references"><div class="references-title">참고 기사:</div>';
        item.related_facts.forEach(fact => {
            const meta = fact.metadata || {};
            const press = meta.press || '언론사 정보 없음';
            const url = meta.url || '#';
            // 제목이 없으면 내용의 앞부분을 사용
            const title = meta.title || (fact.content ? fact.content.substring(0, 30) + '...' : '제목 없음');
            
            referencesHtml += `
                <a href="${url}" target="_blank" class="reference-link">
                    📰 [${press}] ${title}
                </a>
            `;
        });
        referencesHtml += '</div>';
    }

    return `
        <div class="factcheck-item ${typeClass}">
            <div class="factcheck-claim">" ${item.claim} "</div>
            <span class="factcheck-badge ${badgeClass}">${judgment}</span>
            <div class="factcheck-reason">${reason}</div>
            ${referencesHtml}
        </div>
    `;
}
