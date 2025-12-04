/**
 * Fast Naver Factcheck Crawler (Parallel Processing)
 */
require('dotenv').config();
const fs = require('fs/promises');
const cheerio = require('cheerio');
const OpenAI = require('openai');

if (!process.env.OPENAI_API_KEY) {
  console.error('Error: .env 파일에 OPENAI_API_KEY가 없습니다.');
  process.exit(1);
}

const CONFIG = {
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4o-mini', 
  startPage: 1,
  endPage: 6,
  outPath: 'asset/factcheck_ai_summary.txt',
  batchSize: 15, // ★ 핵심: 한 번에 동시에 처리할 기사 수 (너무 높으면 차단 위험)
};

const openai = new OpenAI({ apiKey: CONFIG.apiKey });

function cleanText(text) {
  if (!text) return '';
  return text.replace(/[\n\r\t]+/g, ' ').replace(/\s+/g, ' ').trim();
}

async function summarizeArticle(title, content) {
  try {
    const prompt = `
    다음 뉴스 기사의 '주요 쟁점'과 '팩트체크 결과'를 핵심만 한 문단으로 요약해.
    [제약] 한글 작성, 줄바꿈 금지, 괄호() 사용 금지, 서두 생략.
    [제목] ${title}
    [본문] ${content.substring(0, 2500)}`; // 토큰 절약을 위해 길이 제한

    const completion = await openai.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: CONFIG.model,
    });
    return completion.choices[0].message.content;
  } catch (error) {
    console.error(`  X [AI Error] ${title.substring(0, 10)}... : ${error.message}`);
    return null;
  }
}

// 개별 기사 처리 함수 (병렬 실행용)
async function processArticle(url, title, press, seenSet) {
  if (!url || seenSet.has(url)) return;
  seenSet.add(url);

  try {
    // 1. 본문 가져오기
    const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!res.ok) return;
    
    const html = await res.text();
    const $ = cheerio.load(html);
    const $dic = $('#dic_area');
    $dic.find('script, style, iframe, button, img').remove();
    const bodyText = cleanText($dic.text());

    if (bodyText.length < 50) return;

    // 2. AI 요약 요청 (병렬로 실행됨)
    // console.log(`  > Processing: ${title.substring(0, 15)}...`);
    const summary = await summarizeArticle(title, bodyText);

    if (summary) {
      const cleanSummary = cleanText(summary).replace(/[()]/g, '');
      const shortUrl = url.replace('https://n.news.naver.com/article/', '');
      const line = `[${press}][${shortUrl}] ${cleanSummary}\n`;
      
      // 3. 결과 저장 (비동기 append)
      await fs.appendFile(CONFIG.outPath, line, 'utf8');
      process.stdout.write('.'); // 진행상황 점으로 표시
    }
  } catch (e) {
    console.error(`Error processing ${url}:`, e.message);
  }
}

(async () => {
  try {
    await fs.writeFile(CONFIG.outPath, '', 'utf8');
    console.log(`[Start] Fast Crawling (Batch Size: ${CONFIG.batchSize})`);
    
    const seen = new Set();
    const startTime = Date.now();

    for (let i = CONFIG.startPage; i <= CONFIG.endPage; i++) {
      const listUrl = `https://news.naver.com/factcheck/more?oid=&page=${i}`;
      console.log(`\n[Page ${i}] Fetching list...`);
      
      const res = await fetch(listUrl, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
      });
      if (!res.ok) continue;

      const html = await res.text();
      const $ = cheerio.load(html);
      const cards = $('li.factcheck_card').toArray();

      // --- 병렬 처리 로직 시작 ---
      const tasks = [];
      
      for (const el of cards) {
        const $el = $(el);
        const $link = $el.find('a.factcheck_card_link');
        let articleUrl = $link.attr('href') || '';
        
        if (articleUrl && !/^https?:\/\//i.test(articleUrl)) {
          articleUrl = new URL(articleUrl, 'https://news.naver.com').toString();
        }

        const title = cleanText($link.find('.factcheck_card_title').text());
        const press = cleanText($link.find('.factcheck_card_sub_info .factcheck_card_sub_item').eq(0).text());

        // 작업을 배열에 담음 (아직 실행 X)
        tasks.push(() => processArticle(articleUrl, title, press, seen));
      }

      // 배치 단위로 실행 (예: 5개씩 끊어서 동시 실행)
      for (let j = 0; j < tasks.length; j += CONFIG.batchSize) {
        const batch = tasks.slice(j, j + CONFIG.batchSize);
        // Promise.all로 묶어서 동시 발사
        await Promise.all(batch.map(task => task())); 
      }
      // --- 병렬 처리 로직 끝 ---
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\n[Done] Completed in ${duration}s. Saved to ${CONFIG.outPath}`);

  } catch (err) {
    console.error(err);
  }
})();