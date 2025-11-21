/**
 * data crawling news body from naver news factcheck
 */

// ...existing code...
const fs = require('fs/promises');
const cheerio = require('cheerio');

const outPath = './factcheck_all_pages.json';

function extractKorean(s) {
  if (!s) return '';
  // 한글(가-힣) 및 공백만 남기고 제거, 연속 공백은 하나로 정리
  const parts = s.match(/[가-힣\s]+/g);
  if (!parts) return '';
  return parts.join(' ').replace(/\s+/g, ' ').trim();
}
function clearText(s) {

  let processedText = s;

  //processedText = processedText.replace(/<[^>]*>/g, '');

  //processedText = processedText.replace(/&#\d+;|&nbsp;/g, ' ');



  processedText = processedText
  .trim() 
  .split('\n') 
  .map(line => line.trim()) 
  .filter(line => line.length > 0) 
  .join('\n') 
  .replace(/ +/g, ' '); 
  console.log(processedText,'\n')
  return processedText
}

(async () => {
  try {
    const allCards = [];
    const seen = new Set();

    for (let i = 1; i <= 1; i++) {
      const listUrl = `https://news.naver.com/factcheck/more?oid=&page=${i}`;
      const res = await fetch(listUrl, {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
          Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
      });
      if (!res.ok) throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);

      const html = await res.text();
      await fs.writeFile(`./naver_factcheck_more_page${i}.html`, html, 'utf8');

      const $ = cheerio.load(html);
      const cardEls = $('li.factcheck_card').toArray();

      for (const el of cardEls) {
        const $el = $(el);
        const $link = $el.find('a.factcheck_card_link');
        let articleUrl = $link.attr('href') || '';
        if (articleUrl && !/^https?:\/\//i.test(articleUrl)) {
          articleUrl = new URL(articleUrl, 'https://news.naver.com').toString();
        }

        if (seen.has(articleUrl)) continue;
        seen.add(articleUrl);

        const title = $link.find('.factcheck_card_title').text().trim() || '';
        const desc = $link.find('.factcheck_card_desc').text().trim() || '';
        const subItems = $link.find('.factcheck_card_sub_info .factcheck_card_sub_item');
        const press = subItems.eq(0).text().trim() || '';
        const time = subItems.eq(1).text().trim() || '';
        const img = $link.find('.factcheck_card_img img').attr('src') || '';

        // 개별 기사에서 #dic_area 추출 -> 태그 제거 후 한글만 남김
        let dic_area = '';
        if (articleUrl) {
          try {
            const ares = await fetch(articleUrl, {
              headers: {
                'User-Agent':
                  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
              },
            });
            if (ares.ok) {
              const ahtml = await ares.text();
              const $a = cheerio.load(ahtml);
              const $dic = $a('#dic_area');

              const rawText = $dic.length ? $dic.text() : '';
              dic_area = clearText(rawText);

            } else {
              console.warn(`Article fetch failed: ${articleUrl} -> ${ares.status}`);
            }
          } catch (e) {
            console.warn(`Article fetch error for ${articleUrl}: ${e.message}`);
          }
        }

        allCards.push({ url: articleUrl, title, desc, press, time, img, dic_area });
      }

      console.log(`page ${i} processed, total collected: ${allCards.length}`);
    }

    await fs.writeFile(outPath, JSON.stringify(allCards, null, 2), 'utf8');
    console.log(`Saved merged JSON: ${outPath} (total: ${allCards.length})`);
  } catch (err) {
    console.error('Error:', err);
    process.exit(1);
  }
})();
// ...existing code...