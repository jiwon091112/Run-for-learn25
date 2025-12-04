// 저장된 설정 불러오기
document.addEventListener('DOMContentLoaded', () => {
    chrome.storage.sync.get({
        apiUrl: 'http://localhost:8000' // 기본값
    }, (items) => {
        document.getElementById('apiUrl').value = items.apiUrl;
    });
});

// 설정 저장하기
document.getElementById('save').addEventListener('click', () => {
    const apiUrl = document.getElementById('apiUrl').value.replace(/\/$/, ''); // 끝에 슬래시 제거
    
    chrome.storage.sync.set({
        apiUrl: apiUrl
    }, () => {
        const status = document.getElementById('status');
        status.style.display = 'block';
        setTimeout(() => {
            status.style.display = 'none';
        }, 2000);
    });
});